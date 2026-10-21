import math
import numbers
import random
import warnings
from typing import NamedTuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import is_undirected, negative_sampling

from glpbench.contracts import AuditReport, ProtocolMetadata, SplitBundle
from glpbench.exceptions import ProtocolError
from glpbench.preprocessing import attach_reduced_features, fit_reducer


class QueryNodeSets(NamedTuple):
    """Store disjoint validation and test query-node IDs."""

    validation: tuple[int, ...]
    test: tuple[int, ...]


def _require_successful_audit(report, *, protocol_name, seed):
    """Reject malformed or unsuccessful protocol-audit results."""
    if not isinstance(report, AuditReport):
        raise ProtocolError(
            "protocol audit must return an AuditReport",
            component=protocol_name,
            seed=seed,
        )
    if not report.passed:
        failed = ", ".join(
            name for name, passed in report.checks.items() if not passed
        )
        message = "protocol audit failed"
        if failed:
            message += f": {failed}"
        raise ProtocolError(message, component=protocol_name, seed=seed)
    return report


def _query_selection_metadata(explicit, size, fraction, *, default_mode=None):
    """Describe one query-selection request without storing node IDs."""
    if explicit is not None:
        requested_count = (
            1 if isinstance(explicit, numbers.Integral)
            else int(torch.as_tensor(explicit).numel())
        )
        return {"mode": "explicit", "requested_count": requested_count}
    if size is not None:
        return {"mode": "size", "value": size}
    if fraction is not None:
        return {"mode": "fraction", "value": fraction}
    if default_mode is not None:
        return {"mode": default_mode}
    return {"mode": "unspecified"}


def _labeled_counts(split):
    """Return positive and negative candidate counts for one partition."""
    return (
        int((split.edge_label == 1).sum()),
        int((split.edge_label == 0).sum()),
    )


def _realized_split_counts(
    train,
    validation,
    tests,
    visible_node_mask,
    *,
    validation_queries=None,
    test_queries=None,
    reserved_test_queries=None,
    reserved_test_positives=None,
):
    """Derive the common realized-count vocabulary from PyG partitions."""
    counts = {
        "visible_nodes": int(visible_node_mask.sum()),
        "training_message_passing_edges": int(train.edge_index.size(1)),
    }
    train_positive, train_negative = _labeled_counts(train)
    counts.update({
        "training_positives": train_positive,
        "training_negatives": train_negative,
    })
    if validation is not None:
        positive, negative = _labeled_counts(validation)
        counts.update({
            "validation_message_passing_edges": int(
                validation.edge_index.size(1)
            ),
            "validation_positives": positive,
            "validation_negatives": negative,
        })
    for name, split in tests.items():
        prefix = name.replace("-", "_") + "_test"
        positive, negative = _labeled_counts(split)
        counts.update({
            f"{prefix}_message_passing_edges": int(split.edge_index.size(1)),
            f"{prefix}_positives": positive,
            f"{prefix}_negatives": negative,
        })
    optional_counts = {
        "validation_queries": validation_queries,
        "test_queries": test_queries,
        "reserved_test_queries": reserved_test_queries,
        "reserved_test_positives": reserved_test_positives,
    }
    counts.update({
        name: int(value) for name, value in optional_counts.items()
        if value is not None
    })
    return counts


def _protocol_metadata(
    name,
    seed,
    parameters,
    requested_counts,
    train,
    validation,
    tests,
    visible_node_mask,
    **count_context,
):
    """Build schema-versioned metadata from completed split partitions."""
    return ProtocolMetadata(
        name=name,
        schema_version=2,
        seed=seed,
        parameters=parameters,
        requested_counts=requested_counts,
        realized_counts=_realized_split_counts(
            train,
            validation,
            tests,
            visible_node_mask,
            **count_context,
        ),
    )


def _audit_protocol_metadata(
    splits,
    *,
    expected_name,
    expected_requested_counts,
    expected_realized_counts,
    required_parameters,
):
    """Require protocol metadata to agree with its realized partitions."""
    metadata = splits.metadata
    if metadata.name != expected_name or metadata.schema_version != 2:
        raise ValueError("Protocol metadata name or schema version is invalid.")
    if set(required_parameters).difference(metadata.parameters):
        raise ValueError("Protocol metadata is missing required parameters.")
    if dict(metadata.requested_counts) != dict(expected_requested_counts):
        raise ValueError("Protocol requested counts disagree with the split.")
    if dict(metadata.realized_counts) != dict(expected_realized_counts):
        raise ValueError("Protocol realized counts disagree with the split.")


def _canonical_edge_pairs(edge_index):
    """Return undirected edge pairs as a Python set."""
    return {
        (min(source, target), max(source, target))
        for source, target in edge_index.detach().cpu().t().tolist()
    }


def _validate_simple_undirected_graph(data):
    """Require the simple undirected graph assumed by split construction."""
    if data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    source, target = data.edge_index
    if bool((source == target).any()):
        raise ValueError("Self-loops are not supported by the split protocol.")
    if torch.unique(data.edge_index, dim=1).size(1) != data.edge_index.size(1):
        raise ValueError("Duplicate directed edges are not supported.")
    if not is_undirected(data.edge_index):
        raise NotImplementedError(
            "Split for directed networks is not implemented yet."
        )


def _attach_split_metadata(splits, query_nodes, num_nodes, device):
    """Attach node-role and split-role metadata to graph splits."""
    validation_query_mask = torch.zeros(
        num_nodes, dtype=torch.bool, device=device
    )
    test_query_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    validation_query_mask[list(query_nodes.validation)] = True
    test_query_mask[list(query_nodes.test)] = True
    if bool((validation_query_mask & test_query_mask).any()):
        raise ValueError("Validation and test query nodes must be disjoint.")

    query_mask = validation_query_mask | test_query_mask
    visible_node_mask = ~query_mask
    for _, split_stage, split_setting, split in splits:
        split.validation_query_mask = validation_query_mask.clone()
        split.test_query_mask = test_query_mask.clone()
        split.visible_node_mask = visible_node_mask.clone()

        split.split_stage = split_stage
        split.split_setting = split_setting


def _node_neighbors(data):
    """Return non-self neighbors for every node."""
    neighbors = [set() for _ in range(data.num_nodes)]
    source, target = data.edge_index.detach().cpu().tolist()
    for src, dst in zip(source, target):
        if src != dst:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return neighbors


def _normalize_query_nodes(data, query, name="Query"):
    """Normalize and range-check an explicitly supplied node set."""
    if isinstance(query, bool):
        raise TypeError(f"{name} must be an int or a 1D list or 1D torch tensor.")
    if isinstance(query, int):
        processed_query = [query]
    elif isinstance(query, (list, tuple)):
        if not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in query
        ):
            raise TypeError(
                f"{name} must be an int or a 1D list or 1D torch tensor."
            )
        processed_query = list(query)
    elif isinstance(query, torch.Tensor):
        if query.ndim != 1:
            raise TypeError(
                f"{name} must be an int or a 1D list or 1D torch tensor."
            )
        if query.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise TypeError(f"{name} tensor must use an integer dtype.")
        processed_query = query.detach().cpu().tolist()
    else:
        raise TypeError(f"{name} must be an int or a 1D list or 1D torch tensor.")

    if not processed_query:
        raise ValueError(f"{name} must contain at least one node ID.")
    if any(node < 0 for node in processed_query):
        raise ValueError(f"{name} node IDs must be non-negative integers.")
    if any(node >= data.num_nodes for node in processed_query):
        raise ValueError(f"{name} includes node IDs outside the graph.")
    if len(set(processed_query)) != len(processed_query):
        raise ValueError(f"{name} contains duplicate node IDs.")
    return sorted(processed_query)


def _validate_query_nodes(data, query):
    """Normalize and validate an explicitly supplied query set."""
    processed_query = _normalize_query_nodes(data, query)

    query_nodes = set(processed_query)
    neighbors = _node_neighbors(data)
    invalid = [
        node for node in processed_query
        if not neighbors[node].difference(query_nodes)
    ]
    if invalid:
        raise ValueError(
            "Every query node must have at least one neighbor outside the query "
            f"set; invalid node IDs: {sorted(invalid)}"
        )
    return sorted(processed_query)


def _resolve_query_count(name, size, fraction, num_candidates):
    """Resolve an exact query count from mutually exclusive size options."""
    if (size is None) == (fraction is None):
        raise ValueError(f"Specify exactly one of {name}_size or {name}_fraction.")
    if size is not None:
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(f"{name}_size must be an integer.")
        if size <= 0:
            raise ValueError(f"{name}_size must be a positive integer.")
        resolved_size = size
    else:
        if not isinstance(fraction, numbers.Real) or isinstance(fraction, bool):
            raise TypeError(f"{name}_fraction must be a real number.")
        if not 0 < fraction < 1:
            raise ValueError(f"{name}_fraction must be between 0 and 1.")
        resolved_size = max(
            1, math.floor(fraction * num_candidates + 0.5)
        )
    return resolved_size


def _select_constrained_query_union(
    neighbors, candidates, fixed_nodes, total_count, rng, max_attempts
):
    """Construct a query set while preserving a visible neighbor per query."""
    fixed_nodes = set(fixed_nodes)
    if len(fixed_nodes) > total_count:
        raise ValueError("Fixed query nodes exceed the requested query count.")
    if any(not neighbors[node].difference(fixed_nodes) for node in fixed_nodes):
        raise ValueError(
            "Fixed query nodes do not each retain a training-visible neighbor."
        )

    for _ in range(max_attempts):
        selected = set(fixed_nodes)
        blocked = set()
        for node in selected:
            outside = neighbors[node].difference(selected)
            if len(outside) == 1:
                blocked.update(outside)

        available = [node for node in candidates if node not in selected]
        rng.shuffle(available)
        added = []
        for candidate in available:
            if len(selected) == total_count:
                break
            if candidate in blocked:
                continue
            candidate_outside = neighbors[candidate].difference(selected)
            candidate_outside.discard(candidate)
            if not candidate_outside:
                continue

            newly_blocked = set()
            valid = True
            for selected_neighbor in neighbors[candidate].intersection(selected):
                outside = neighbors[selected_neighbor].difference(selected)
                if len(outside) <= 1:
                    valid = False
                    break
                if len(outside) == 2:
                    newly_blocked.update(outside.difference({candidate}))
            if not valid:
                continue

            selected.add(candidate)
            added.append(candidate)
            blocked.update(newly_blocked)
            remaining = neighbors[candidate].difference(selected)
            if len(remaining) == 1:
                blocked.update(remaining)

        if len(selected) == total_count:
            return selected, added

    raise ValueError(
        "Could not construct disjoint validation and test query sets for which "
        "every query retains a visible neighbor outside the query set after "
        f"{max_attempts} attempts."
    )


def select_query_node_sets(
    data, *, test_query=None, validation_query=None, query_size=None,
    query_fraction=None, validation_query_size=None,
    validation_query_fraction=None, seed=None, max_attempts=1000,
):
    """Select disjoint validation and test query-node sets.

    Every selected query retains at least one neighbor outside the union of
    both query sets. If validation sizing is omitted, its node count matches
    the resolved test-query count.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Simple undirected source graph.
    test_query, validation_query : int or sequence of int, optional
        Explicit test or validation query-node IDs.
    query_size, validation_query_size : int, optional
        Exact automatically selected query counts.
    query_fraction, validation_query_fraction : float, optional
        Fractions of eligible non-isolated nodes selected as queries.
    seed : int, optional
        Seed for the local query-selection generator.
    max_attempts : int, optional
        Maximum randomized construction attempts.

    Returns
    -------
    QueryNodeSets
        Sorted, disjoint validation and test query-node IDs.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError(
            "Query selection for heterogeneous data is not implemented."
        )
    if not is_undirected(data.edge_index):
        raise NotImplementedError(
            "Query selection for directed networks is not implemented."
        )
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts must be a positive integer.")

    neighbors = _node_neighbors(data)
    candidates = sorted(node for node, adjacent in enumerate(neighbors) if adjacent)
    if not candidates:
        raise ValueError(
            "Cannot select query nodes from a graph without non-self edges."
        )

    if test_query is not None:
        if query_size is not None or query_fraction is not None:
            raise ValueError(
                "Explicit test queries cannot be combined with query_size or "
                "query_fraction."
            )
        fixed_test = _normalize_query_nodes(data, test_query, "Test query")
        test_count = len(fixed_test)
    else:
        fixed_test = None
        test_count = _resolve_query_count(
            "query", query_size, query_fraction, len(candidates)
        )

    if validation_query is not None:
        if (
            validation_query_size is not None
            or validation_query_fraction is not None
        ):
            raise ValueError(
                "Explicit validation queries cannot be combined with "
                "validation query sizing."
            )
        fixed_validation = _normalize_query_nodes(
            data, validation_query, "Validation query"
        )
        validation_count = len(fixed_validation)
    else:
        fixed_validation = None
        if (
            validation_query_size is None
            and validation_query_fraction is None
        ):
            validation_count = test_count
        else:
            validation_count = _resolve_query_count(
                "validation_query",
                validation_query_size,
                validation_query_fraction,
                len(candidates),
            )

    fixed_test_set = set(fixed_test or ())
    fixed_validation_set = set(fixed_validation or ())
    overlap = fixed_test_set & fixed_validation_set
    if overlap:
        raise ValueError(
            "Validation and test query nodes must be disjoint; overlapping "
            f"node IDs: {sorted(overlap)}"
        )
    total_count = test_count + validation_count
    if total_count >= data.num_nodes or total_count > len(candidates):
        raise ValueError(
            "Requested validation and test query counts leave no feasible "
            "training-visible node partition."
        )

    rng = random.Random(seed)
    missing_test = test_count - len(fixed_test_set)
    missing_validation = validation_count - len(fixed_validation_set)
    fixed_union = fixed_test_set | fixed_validation_set
    if missing_test + missing_validation > len(set(candidates) - fixed_union):
        raise ValueError(
            "Requested validation and test queries exceed available candidates."
        )
    _, added = _select_constrained_query_union(
        neighbors,
        candidates,
        fixed_union,
        total_count,
        rng,
        max_attempts,
    )
    test_nodes = sorted(fixed_test_set | set(added[:missing_test]))
    validation_nodes = sorted(
        fixed_validation_set | set(added[missing_test:])
    )
    return QueryNodeSets(
        validation=tuple(validation_nodes), test=tuple(test_nodes)
    )


def select_query_nodes(
    data, *, query_size=None, query_fraction=None, seed=None, max_attempts=1000
):
    """Select a reproducible set of cold-start query nodes.

    Each selected node has at least one neighbor outside the selected set. The
    function uses a local random generator and does not modify global RNG state.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Undirected source graph.
    query_size : int, optional
        Exact number of query nodes.
    query_fraction : float, optional
        Fraction of eligible, non-isolated nodes selected as queries.
    seed : int, optional
        Random seed used only for query selection.
    max_attempts : int, optional
        Maximum candidate sets tested before reporting infeasibility.

    Returns
    -------
    list of int
        Sorted query-node IDs.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError(
            "Query selection for heterogeneous data is not implemented."
        )
    if not is_undirected(data.edge_index):
        raise NotImplementedError(
            "Query selection for directed networks is not implemented."
        )
    if (query_size is None) == (query_fraction is None):
        raise ValueError("Specify exactly one of query_size or query_fraction.")
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts <= 0
    ):
        raise ValueError("max_attempts must be a positive integer.")

    neighbors = _node_neighbors(data)
    candidates = sorted(node for node, adjacent in enumerate(neighbors) if adjacent)
    if not candidates:
        raise ValueError(
            "Cannot select query nodes from a graph without non-self edges."
        )

    if query_size is not None:
        if not isinstance(query_size, int) or isinstance(query_size, bool):
            raise TypeError("query_size must be an integer.")
        if query_size <= 0:
            raise ValueError("query_size must be a positive integer.")
        resolved_size = query_size
    else:
        if (
            not isinstance(query_fraction, numbers.Real)
            or isinstance(query_fraction, bool)
        ):
            raise TypeError("query_fraction must be a real number.")
        if not 0 < query_fraction < 1:
            raise ValueError("query_fraction must be between 0 and 1.")
        resolved_size = max(1, math.floor(query_fraction * len(candidates) + 0.5))

    if resolved_size > len(candidates):
        raise ValueError(
            "Requested query count exceeds the number of eligible "
            "non-isolated nodes."
        )

    rng = random.Random(seed)
    selected, _ = _select_constrained_query_union(
        neighbors,
        candidates,
        fixed_nodes=(),
        total_count=resolved_size,
        rng=rng,
        max_attempts=max_attempts,
    )
    return sorted(selected)


def _resolve_query_nodes(
    data, query, *, query_size=None, query_fraction=None, seed=None,
    test_ratio=None
):
    """Resolve explicit or automatically selected query nodes."""
    if test_ratio is not None:
        if query_fraction is not None:
            raise ValueError("Specify only one of test_ratio and query_fraction.")
        warnings.warn(
            "test_ratio is deprecated; use query_fraction instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        query_fraction = test_ratio

    if query is not None:
        if query_size is not None or query_fraction is not None:
            raise ValueError(
                "Explicit query nodes cannot be combined with query_size or "
                "query_fraction."
            )
        return _validate_query_nodes(data, query)

    return select_query_nodes(
        data,
        query_size=query_size,
        query_fraction=query_fraction,
        seed=seed,
    )


def _resolve_query_node_sets(
    data, query, validation_query, *, query_size=None, query_fraction=None,
    validation_query_size=None, validation_query_fraction=None, seed=None,
    test_ratio=None,
):
    """Resolve disjoint validation and test query sets."""
    if test_ratio is not None:
        if query_fraction is not None:
            raise ValueError("Specify only one of test_ratio and query_fraction.")
        warnings.warn(
            "test_ratio is deprecated; use query_fraction instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        query_fraction = test_ratio

    return select_query_node_sets(
        data,
        test_query=query,
        validation_query=validation_query,
        query_size=query_size,
        query_fraction=query_fraction,
        validation_query_size=validation_query_size,
        validation_query_fraction=validation_query_fraction,
        seed=seed,
    )


def _sample_inductive_negative_edges(data, positive_edges, query, seed=None):
    """Sample non-edges targeting the positive edges' query nodes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Original graph.
    positive_edges : torch.Tensor
        Positive edges with shape ``[2, num_edges]``.
    query : sequence of int
        Query node IDs excluded from candidate sources.
    seed : int, optional
        Local random seed.

    Returns
    -------
    torch.Tensor
        Negative edges with shape ``[2, num_edges]``.
    """
    edge_dtype = data.edge_index.dtype
    edge_device = data.edge_index.device
    query_nodes = set(query)

    neighbors = [set() for _ in range(data.num_nodes)]
    src_nodes, tgt_nodes = data.edge_index.detach().cpu().tolist()
    for src, tgt in zip(src_nodes, tgt_nodes):
        neighbors[src].add(tgt)
        neighbors[tgt].add(src)

    rng = random.Random(seed)
    sources_by_target = {}
    negative_targets = positive_edges[1].detach().cpu().tolist()
    for target in set(negative_targets):
        required = negative_targets.count(target)
        candidates = [
            node for node in range(data.num_nodes)
            if node not in query_nodes
            and node != target
            and node not in neighbors[target]
        ]
        if len(candidates) < required:
            raise ValueError(
                f"Cannot sample a negative edge for query node {target}: "
                f"{required} unique negatives are required but only "
                f"{len(candidates)} eligible non-neighbors are available; "
                "no eligible non-neighbor can be reused."
            )
        sources_by_target[target] = iter(rng.sample(candidates, required))

    negative_sources = [
        next(sources_by_target[target]) for target in negative_targets
    ]

    return torch.tensor(
        [negative_sources, negative_targets],
        dtype=edge_dtype,
        device=edge_device,
    )


def _sample_seen_negative_edges(data, seen_node_mask, num_samples, seed=None):
    """Sample unique undirected non-edges between visible nodes.

    Candidates are checked against the original graph in ``data``. A local
    node-ID mapping prevents query nodes from entering the sampling universe.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Original undirected graph containing every known positive edge.
    seen_node_mask : torch.Tensor
        Boolean mask identifying training-visible nodes.
    num_samples : int
        Number of undirected negative edges to sample.
    seed : int, optional
        Local random seed that does not modify global PyTorch RNG state.

    Returns
    -------
    torch.Tensor
        Negative edges with shape ``[2, num_samples]`` and one canonical
        orientation per undirected pair.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if not is_undirected(data.edge_index):
        raise NotImplementedError(
            "Seen-node negative sampling requires an undirected graph."
        )
    if not isinstance(seen_node_mask, torch.Tensor):
        raise TypeError("seen_node_mask must be a torch.Tensor.")
    if seen_node_mask.dtype != torch.bool:
        raise TypeError("seen_node_mask must use boolean dtype.")
    if seen_node_mask.ndim != 1 or seen_node_mask.numel() != data.num_nodes:
        raise ValueError("seen_node_mask must contain one entry per node.")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")

    device = data.edge_index.device
    seen_node_mask = seen_node_mask.to(device=device)
    seen_node_ids = seen_node_mask.nonzero(as_tuple=False).view(-1)
    edge_dtype = data.edge_index.dtype
    if num_samples == 0:
        return torch.empty((2, 0), dtype=edge_dtype, device=device)
    if seen_node_ids.numel() < 2:
        raise ValueError("At least two visible nodes are required.")

    global_to_seen = torch.full(
        (data.num_nodes,), -1, dtype=torch.long, device=device
    )
    global_to_seen[seen_node_ids] = torch.arange(
        seen_node_ids.numel(), device=device
    )

    source, target = data.edge_index
    visible_edge_mask = (
        seen_node_mask[source]
        & seen_node_mask[target]
        & (source != target)
    )
    local_visible_edges = global_to_seen[data.edge_index[:, visible_edge_mask]]

    if local_visible_edges.numel() == 0:
        canonical_positives = torch.empty(
            (2, 0), dtype=torch.long, device=device
        )
    else:
        canonical_positives = torch.unique(
            torch.sort(local_visible_edges, dim=0).values,
            dim=1,
        )

    num_seen = seen_node_ids.numel()
    num_possible_pairs = num_seen * (num_seen - 1) // 2
    num_available = num_possible_pairs - canonical_positives.size(1)
    if num_samples > num_available:
        raise ValueError(
            f"Cannot sample {num_samples} visible-node negative edges: only "
            f"{num_available} eligible non-edges are available."
        )

    def sample_negatives():
        return negative_sampling(
            edge_index=local_visible_edges,
            num_nodes=num_seen,
            num_neg_samples=2 * num_samples,
            method="sparse",
            force_undirected=True,
        )

    if seed is None:
        bidirectional_negatives = sample_negatives()
    else:
        devices = []
        if device.type == "cuda":
            devices = [device.index]
        python_random_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                torch.manual_seed(seed)
                bidirectional_negatives = sample_negatives()
        finally:
            random.setstate(python_random_state)

    local_negatives = bidirectional_negatives[
        :, bidirectional_negatives[0] < bidirectional_negatives[1]
    ]
    local_negatives = torch.unique(local_negatives, dim=1)
    if local_negatives.size(1) != num_samples:
        raise RuntimeError(
            "Negative sampling did not return the requested number of unique "
            "undirected edges."
        )

    return seen_node_ids[local_negatives].to(dtype=edge_dtype)


def _set_negative_edges(split, negative_edges):
    """Append negative edges and rebuild binary edge labels."""
    positive_edges = split.edge_label_index
    split.edge_label_index = torch.cat(
        [positive_edges, negative_edges], dim=1
    )
    split.edge_label = torch.cat([
        split.edge_label.new_ones(positive_edges.size(1)),
        split.edge_label.new_zeros(negative_edges.size(1)),
    ])


def _visible_node_mask(data, query_nodes):
    """Return the mask of nodes visible during model fitting."""
    mask = torch.ones(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    mask[list(query_nodes.validation)] = False
    mask[list(query_nodes.test)] = False
    return mask


def _query_positive_edges(data, query, visible_node_mask):
    """Return original edges oriented from visible nodes to query nodes."""
    query_mask = torch.zeros_like(visible_node_mask)
    query_mask[list(query)] = True
    source, target = data.edge_index
    return data.edge_index[
        :, visible_node_mask[source] & query_mask[target]
    ]


def _semi_inductive_split_data(
    data, message_passing_edges, positive_edges, all_queries, seed
):
    """Build a labeled seen-to-query evaluation split."""
    split = data.clone()
    split.edge_index = message_passing_edges
    split.edge_label_index = positive_edges
    split.edge_label = torch.ones(
        positive_edges.size(1), device=data.edge_index.device
    )
    negative_edges = _sample_inductive_negative_edges(
        data, positive_edges, all_queries, seed=seed
    )
    _set_negative_edges(split, negative_edges)
    return split


def _apply_random_link_split(data, num_val, num_test, seed):
    """Apply ``RandomLinkSplit`` without modifying global RNG state."""
    splitter = T.RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        add_negative_train_samples=False,
        is_undirected=True,
        neg_sampling_ratio=0.0,
    )
    if seed is None:
        return splitter(data)

    devices = []
    if data.edge_index.device.type == "cuda":
        devices = [data.edge_index.device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return splitter(data)


def _audit_comparable_splits(data, splits):
    """Raise ``ValueError`` if a comparable split violates leakage rules.

    The audit covers node roles, message-passing topology, positive-set
    disjointness, negative-edge validity, endpoint orientation, and matching
    of the paired test sets.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Original simple undirected graph.
    splits : SplitBundle
        Splits produced by :func:`generate_comparable_splits`.

    Returns
    -------
    None
        Returns only after every invariant passes.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if not isinstance(splits, SplitBundle):
        raise TypeError("splits should be a SplitBundle")
    if splits.validation is None or set(splits.tests) != {
        "transductive", "semi-inductive"
    }:
        raise ValueError("comparable splits require both paired test settings")
    _validate_simple_undirected_graph(data)

    split_items = (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test_transductive", splits.tests["transductive"]),
        ("test_inductive", splits.tests["semi-inductive"]),
    )
    for name, split in split_items:
        if split.num_nodes != data.num_nodes:
            raise ValueError(f"{name} has a different number of nodes.")
        if split.edge_label_index.ndim != 2 or split.edge_label_index.size(0) != 2:
            raise ValueError(f"{name} edge_label_index must have shape [2, E].")
        if split.edge_label.numel() != split.edge_label_index.size(1):
            raise ValueError(f"{name} edge labels and labeled edges differ in size.")
        if not bool(((split.edge_label == 0) | (split.edge_label == 1)).all()):
            raise ValueError(f"{name} edge labels must be binary.")

    reference = splits.train
    required_masks = (
        "validation_query_mask", "test_query_mask", "visible_node_mask",
    )
    for attribute in required_masks:
        expected = getattr(reference, attribute, None)
        if (
            not isinstance(expected, torch.Tensor)
            or expected.dtype != torch.bool
            or expected.ndim != 1
            or expected.numel() != data.num_nodes
        ):
            raise ValueError(f"Invalid or missing {attribute} metadata.")
        for name, split in split_items[1:]:
            if not torch.equal(getattr(split, attribute, None), expected):
                raise ValueError(f"{name} has inconsistent {attribute} metadata.")

    validation_mask = reference.validation_query_mask
    test_mask = reference.test_query_mask
    visible_mask = reference.visible_node_mask
    if bool((validation_mask & test_mask).any()):
        raise ValueError("Validation and test query sets overlap.")
    if not torch.equal(visible_mask, ~(validation_mask | test_mask)):
        raise ValueError("Visible-node metadata is inconsistent with query sets.")

    original_pairs = _canonical_edge_pairs(data.edge_index)
    message_passing_pairs = {}
    positive_pairs = {}
    negative_pairs = {}
    for name, split in split_items:
        if split.edge_index.numel() and not bool(visible_mask[split.edge_index].all()):
            raise ValueError(f"{name} message-passing edges contain a query node.")
        if not is_undirected(split.edge_index):
            raise ValueError(f"{name} message-passing graph is not undirected.")
        message_passing_pairs[name] = _canonical_edge_pairs(split.edge_index)
        if not message_passing_pairs[name].issubset(original_pairs):
            raise ValueError(f"{name} message-passing graph contains a non-edge.")

        positives = split.edge_label_index[:, split.edge_label == 1]
        negatives = split.edge_label_index[:, split.edge_label == 0]
        positive_pairs[name] = _canonical_edge_pairs(positives)
        negative_pairs[name] = _canonical_edge_pairs(negatives)
        if len(positive_pairs[name]) != positives.size(1):
            raise ValueError(f"{name} contains duplicate positive pairs.")
        if len(negative_pairs[name]) != negatives.size(1):
            raise ValueError(f"{name} contains duplicate negative pairs.")
        if not positive_pairs[name].issubset(original_pairs):
            raise ValueError(f"{name} contains a positive edge absent from the graph.")
        if negative_pairs[name] & original_pairs:
            raise ValueError(f"{name} contains a negative edge that is positive.")
        if any(source == target for source, target in negative_pairs[name]):
            raise ValueError(f"{name} contains a negative self-loop.")

    if not torch.equal(splits.validation.edge_index, splits.train.edge_index):
        raise ValueError("Training and validation message-passing graphs differ.")
    if not torch.equal(
        splits.tests["transductive"].edge_index,
        splits.tests["semi-inductive"].edge_index,
    ):
        raise ValueError("The paired test message-passing graphs differ.")
    if (
        splits.tests["transductive"].edge_label_index.size(1)
        != splits.tests["semi-inductive"].edge_label_index.size(1)
    ):
        raise ValueError("The paired test sets have different labeled-edge counts.")

    for name in ("validation", "test_transductive", "test_inductive"):
        if positive_pairs[name] & message_passing_pairs[name]:
            raise ValueError(
                f"{name} positives overlap its message-passing graph."
            )
    positive_names = tuple(positive_pairs)
    for index, left in enumerate(positive_names):
        for right in positive_names[index + 1:]:
            if positive_pairs[left] & positive_pairs[right]:
                raise ValueError(
                    f"Positive edges overlap between {left} and {right}."
                )

    role_checks = (
        ("validation", splits.validation, validation_mask),
        ("test_inductive", splits.tests["semi-inductive"], test_mask),
    )
    for name, split, target_mask in role_checks:
        sources, targets = split.edge_label_index
        if split.edge_label_index.numel() and (
            not bool(visible_mask[sources].all())
            or not bool(target_mask[targets].all())
        ):
            raise ValueError(
                f"{name} edges must be oriented from visible sources to queries."
            )
        num_positive = int((split.edge_label == 1).sum())
        num_negative = int((split.edge_label == 0).sum())
        if num_positive != num_negative:
            raise ValueError(f"{name} does not have a 1:1 negative ratio.")

    transductive = splits.tests["transductive"]
    transductive_nodes = transductive.edge_label_index
    if transductive_nodes.numel() and not bool(visible_mask[transductive_nodes].all()):
        raise ValueError("Transductive labeled edges contain a query node.")
    if int((transductive.edge_label == 1).sum()) != int(
        (transductive.edge_label == 0).sum()
    ):
        raise ValueError("The transductive test does not have a 1:1 negative ratio.")

    query_mask = validation_mask | test_mask
    query_query_pairs = {
        pair for pair in original_pairs
        if bool(query_mask[pair[0]]) and bool(query_mask[pair[1]])
    }
    represented_pairs = set().union(
        *message_passing_pairs.values(),
        *positive_pairs.values(),
        *negative_pairs.values(),
    )
    if query_query_pairs & represented_pairs:
        raise ValueError("Query-to-query edges must be excluded from all splits.")

    _audit_protocol_metadata(
        splits,
        expected_name="matched-semi-inductive",
        expected_requested_counts={
            "validation_queries": len(splits.validation_query_nodes),
            "test_queries": len(splits.test_query_nodes),
        },
        expected_realized_counts=_realized_split_counts(
            splits.train,
            splits.validation,
            splits.tests,
            visible_mask,
            validation_queries=len(splits.validation_query_nodes),
            test_queries=len(splits.test_query_nodes),
        ),
        required_parameters={
            "train_ratio", "negative_sampling_ratio",
            "test_query_selection", "validation_query_selection",
        },
    )

    return AuditReport(passed=True, checks={"all-invariants": True})


def audit_comparable_splits(data, splits):
    """Audit a paired split and raise ``ProtocolError`` on violations."""
    try:
        return _audit_comparable_splits(data, splits)
    except ValueError as error:
        raise ProtocolError(
            str(error),
            component="matched-semi-inductive",
            seed=(
                splits.metadata.seed
                if isinstance(splits, SplitBundle)
                else None
            ),
        ) from error


def _audit_semi_inductive_bundle(data, splits, *, test_key=None):
    """Validate shared invariants for semi-inductive split bundles."""
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if not isinstance(splits, SplitBundle):
        raise TypeError("splits should be a SplitBundle")
    _validate_simple_undirected_graph(data)
    if splits.validation is None:
        raise ValueError("semi-inductive protocols require validation data")
    expected_tests = set() if test_key is None else {test_key}
    if set(splits.tests) != expected_tests:
        raise ValueError("split bundle contains unexpected test partitions")

    split_items = [("train", splits.train), ("validation", splits.validation)]
    if test_key is not None:
        split_items.append(("test", splits.tests[test_key]))
    required_masks = (
        "validation_query_mask", "test_query_mask", "visible_node_mask",
    )
    for attribute in required_masks:
        expected = getattr(splits.train, attribute, None)
        if (
            not isinstance(expected, torch.Tensor)
            or expected.dtype != torch.bool
            or expected.ndim != 1
            or expected.numel() != data.num_nodes
        ):
            raise ValueError(f"Invalid or missing {attribute} metadata.")
        for name, split in split_items[1:]:
            if not torch.equal(getattr(split, attribute, None), expected):
                raise ValueError(f"{name} has inconsistent {attribute} metadata.")

    validation_mask = splits.train.validation_query_mask
    test_mask = splits.train.test_query_mask
    visible_mask = splits.train.visible_node_mask
    if bool((validation_mask & test_mask).any()):
        raise ValueError("Validation and test query sets overlap.")
    if not torch.equal(visible_mask, ~(validation_mask | test_mask)):
        raise ValueError("Visible-node metadata is inconsistent with query sets.")
    if set(validation_mask.nonzero().view(-1).tolist()) != set(
        splits.validation_query_nodes
    ):
        raise ValueError("Validation-query mask disagrees with SplitBundle metadata.")
    if set(test_mask.nonzero().view(-1).tolist()) != set(
        splits.test_query_nodes
    ):
        raise ValueError("Test-query mask disagrees with SplitBundle metadata.")

    original_pairs = _canonical_edge_pairs(data.edge_index)
    positive_pairs = {}
    represented_pairs = set()
    for name, split in split_items:
        if split.num_nodes != data.num_nodes:
            raise ValueError(f"{name} has a different number of nodes.")
        if split.edge_index.numel() and not bool(visible_mask[split.edge_index].all()):
            raise ValueError(f"{name} message-passing edges contain a query node.")
        if not is_undirected(split.edge_index):
            raise ValueError(f"{name} message-passing graph is not undirected.")
        message_pairs = _canonical_edge_pairs(split.edge_index)
        if not message_pairs.issubset(original_pairs):
            raise ValueError(f"{name} message-passing graph contains a non-edge.")
        represented_pairs.update(message_pairs)
        if split.edge_label_index.ndim != 2 or split.edge_label_index.size(0) != 2:
            raise ValueError(f"{name} edge_label_index must have shape [2, E].")
        if split.edge_label.numel() != split.edge_label_index.size(1):
            raise ValueError(f"{name} edge labels and labeled edges differ in size.")
        if not bool(((split.edge_label == 0) | (split.edge_label == 1)).all()):
            raise ValueError(f"{name} edge labels must be binary.")
        positives = split.edge_label_index[:, split.edge_label == 1]
        negatives = split.edge_label_index[:, split.edge_label == 0]
        positive_pairs[name] = _canonical_edge_pairs(positives)
        negative_pairs = _canonical_edge_pairs(negatives)
        if len(positive_pairs[name]) != positives.size(1):
            raise ValueError(f"{name} contains duplicate positive pairs.")
        if len(negative_pairs) != negatives.size(1):
            raise ValueError(f"{name} contains duplicate negative pairs.")
        if not positive_pairs[name].issubset(original_pairs):
            raise ValueError(f"{name} contains a positive edge absent from the graph.")
        if negative_pairs & original_pairs:
            raise ValueError(f"{name} contains a negative edge that is positive.")
        represented_pairs.update(positive_pairs[name])
        represented_pairs.update(negative_pairs)
        if name != "train" and positive_pairs[name] & message_pairs:
            raise ValueError(f"{name} positives overlap its message-passing graph.")
        if name != "train" and positives.size(1) != negatives.size(1):
            raise ValueError(f"{name} does not have a 1:1 negative ratio.")

    if not torch.equal(splits.validation.edge_index, splits.train.edge_index):
        raise ValueError("Training and validation message-passing graphs differ.")
    positive_names = tuple(positive_pairs)
    for index, left in enumerate(positive_names):
        for right in positive_names[index + 1:]:
            if positive_pairs[left] & positive_pairs[right]:
                raise ValueError(f"Positive edges overlap between {left} and {right}.")

    roles = [("validation", splits.validation, validation_mask)]
    if test_key is not None:
        roles.append(("test", splits.tests[test_key], test_mask))
    for name, split, target_mask in roles:
        sources, targets = split.edge_label_index
        if split.edge_label_index.numel() and (
            not bool(visible_mask[sources].all())
            or not bool(target_mask[targets].all())
        ):
            raise ValueError(
                f"{name} edges must be oriented from visible sources to queries."
            )

    query_mask = validation_mask | test_mask
    query_query_pairs = {
        pair for pair in original_pairs
        if bool(query_mask[pair[0]]) and bool(query_mask[pair[1]])
    }
    if query_query_pairs & represented_pairs:
        raise ValueError("Query-to-query edges must be excluded from all splits.")

    if test_key is None:
        reserved_test_positives = _query_positive_edges(
            data, splits.test_query_nodes, visible_mask
        ).size(1)
        expected_name = "matched-semi-inductive-validation-only"
        expected_requested = {
            "validation_queries": len(splits.validation_query_nodes),
            "reserved_test_queries": len(splits.test_query_nodes),
            "reserved_test_positives": reserved_test_positives,
        }
        count_context = {
            "validation_queries": len(splits.validation_query_nodes),
            "reserved_test_queries": len(splits.test_query_nodes),
            "reserved_test_positives": reserved_test_positives,
        }
        required_parameters = {
            "train_ratio", "negative_sampling_ratio", "reserve_test_capacity",
            "test_query_selection", "validation_query_selection",
        }
    else:
        expected_name = "semi-inductive"
        expected_requested = {
            "validation_queries": len(splits.validation_query_nodes),
            "test_queries": len(splits.test_query_nodes),
        }
        count_context = {
            "validation_queries": len(splits.validation_query_nodes),
            "test_queries": len(splits.test_query_nodes),
        }
        required_parameters = {
            "train_ratio", "negative_sampling_ratio",
            "test_query_selection", "validation_query_selection",
        }
    _audit_protocol_metadata(
        splits,
        expected_name=expected_name,
        expected_requested_counts=expected_requested,
        expected_realized_counts=_realized_split_counts(
            splits.train,
            splits.validation,
            splits.tests,
            visible_mask,
            **count_context,
        ),
        required_parameters=required_parameters,
    )
    return AuditReport(passed=True, checks={"all-invariants": True})


def audit_validation_splits(data, splits):
    """Audit a validation-only semi-inductive split bundle."""
    try:
        return _audit_semi_inductive_bundle(data, splits)
    except ValueError as error:
        raise ProtocolError(
            str(error),
            component="matched-semi-inductive-validation-only",
            seed=splits.metadata.seed if isinstance(splits, SplitBundle) else None,
        ) from error


def audit_semi_inductive_splits(data, splits):
    """Audit a standalone semi-inductive split bundle."""
    try:
        return _audit_semi_inductive_bundle(
            data, splits, test_key="semi-inductive"
        )
    except ValueError as error:
        raise ProtocolError(
            str(error),
            component="semi-inductive",
            seed=splits.metadata.seed if isinstance(splits, SplitBundle) else None,
        ) from error


def _audit_transductive_splits(data, splits):
    """Validate standalone transductive split invariants."""
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if not isinstance(splits, SplitBundle):
        raise TypeError("splits should be a SplitBundle")
    _validate_simple_undirected_graph(data)
    if splits.validation is None or set(splits.tests) != {"transductive"}:
        raise ValueError("transductive protocol requires validation and test data")
    if splits.validation_query_nodes or splits.test_query_nodes:
        raise ValueError("transductive protocols cannot contain query nodes")
    visible_mask = splits.visible_node_mask
    if (
        not isinstance(visible_mask, torch.Tensor)
        or visible_mask.dtype != torch.bool
        or visible_mask.numel() != data.num_nodes
        or not bool(visible_mask.all())
    ):
        raise ValueError("transductive protocol must mark every node visible")

    original_pairs = _canonical_edge_pairs(data.edge_index)
    split_items = (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.tests["transductive"]),
    )
    positive_pairs = {}
    for name, split in split_items:
        for attribute in (
            "validation_query_mask", "test_query_mask", "visible_node_mask",
        ):
            mask = getattr(split, attribute, None)
            if not isinstance(mask, torch.Tensor) or not torch.equal(
                mask, getattr(splits.train, attribute)
            ):
                raise ValueError(f"{name} has inconsistent {attribute} metadata")
        if split.num_nodes != data.num_nodes or not is_undirected(split.edge_index):
            raise ValueError(f"{name} has an invalid message-passing graph")
        message_pairs = _canonical_edge_pairs(split.edge_index)
        if not message_pairs.issubset(original_pairs):
            raise ValueError(f"{name} message-passing graph contains a non-edge")
        if split.edge_label_index.ndim != 2 or split.edge_label_index.size(0) != 2:
            raise ValueError(f"{name} edge_label_index must have shape [2, E]")
        if split.edge_label.numel() != split.edge_label_index.size(1):
            raise ValueError(f"{name} edge labels and labeled edges differ in size")
        positives = split.edge_label_index[:, split.edge_label == 1]
        negatives = split.edge_label_index[:, split.edge_label == 0]
        positive_pairs[name] = _canonical_edge_pairs(positives)
        negative_pairs = _canonical_edge_pairs(negatives)
        if len(positive_pairs[name]) != positives.size(1):
            raise ValueError(f"{name} contains duplicate positive pairs")
        if len(negative_pairs) != negatives.size(1):
            raise ValueError(f"{name} contains duplicate negative pairs")
        if not positive_pairs[name].issubset(original_pairs):
            raise ValueError(f"{name} contains a positive edge absent from the graph")
        if negative_pairs & original_pairs:
            raise ValueError(f"{name} contains a negative edge that is positive")
        if name != "train" and positive_pairs[name] & message_pairs:
            raise ValueError(f"{name} positives overlap its message-passing graph")
        if name != "train" and positives.size(1) != negatives.size(1):
            raise ValueError(f"{name} does not have a 1:1 negative ratio")
    names = tuple(positive_pairs)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if positive_pairs[left] & positive_pairs[right]:
                raise ValueError(f"Positive edges overlap between {left} and {right}")
    _audit_protocol_metadata(
        splits,
        expected_name="transductive",
        expected_requested_counts={
            "validation_positives": int(
                splits.metadata.parameters["validation_ratio"]
                * (data.edge_index[0] < data.edge_index[1]).sum()
            ),
            "test_positives": int(
                splits.metadata.parameters["test_ratio"]
                * (data.edge_index[0] < data.edge_index[1]).sum()
            ),
        },
        expected_realized_counts=_realized_split_counts(
            splits.train,
            splits.validation,
            splits.tests,
            visible_mask,
        ),
        required_parameters={
            "validation_ratio", "test_ratio", "negative_sampling_ratio",
        },
    )
    return AuditReport(passed=True, checks={"all-invariants": True})


def audit_transductive_splits(data, splits):
    """Audit a standalone transductive split bundle."""
    try:
        return _audit_transductive_splits(data, splits)
    except ValueError as error:
        raise ProtocolError(
            str(error),
            component="transductive",
            seed=splits.metadata.seed if isinstance(splits, SplitBundle) else None,
        ) from error


def generate_transductive_splits(
    data, *, validation_ratio=0.1, test_ratio=0.1, seed=None
):
    """Generate standalone random-edge transductive splits.

    Validation and test positives are withheld by PyG's ``RandomLinkSplit``.
    Evaluation negatives are then sampled at a 1:1 ratio from non-edges in the
    original graph. Training negatives remain the responsibility of the
    training loop.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    _validate_simple_undirected_graph(data)
    for value, name in (
        (validation_ratio, "validation_ratio"),
        (test_ratio, "test_ratio"),
    ):
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not 0 <= value < 1
        ):
            raise ValueError(f"{name} must be between zero and one")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must sum to less than one")

    train, validation, test = _apply_random_link_split(
        data.clone(), validation_ratio, test_ratio, seed
    )
    visible_mask = torch.ones(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    for offset, split in enumerate((validation, test), start=1):
        negative_edges = _sample_seen_negative_edges(
            data,
            visible_mask,
            int(split.edge_label_index.size(1)),
            seed=None if seed is None else seed + offset,
        )
        _set_negative_edges(split, negative_edges)
    query_nodes = QueryNodeSets(validation=(), test=())
    _attach_split_metadata(
        (
            ("train", "train", "transductive", train),
            ("validation", "validation", "transductive", validation),
            ("transductive", "test", "transductive", test),
        ),
        query_nodes,
        data.num_nodes,
        data.edge_index.device,
    )
    splits = SplitBundle(
        train=train,
        validation=validation,
        tests={"transductive": test},
        metadata=_protocol_metadata(
            "transductive",
            seed,
            parameters={
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "negative_sampling_ratio": 1.0,
            },
            requested_counts={
                "validation_positives": int(
                    validation_ratio * (data.edge_index[0] < data.edge_index[1]).sum()
                ),
                "test_positives": int(
                    test_ratio * (data.edge_index[0] < data.edge_index[1]).sum()
                ),
            },
            train=train,
            validation=validation,
            tests={"transductive": test},
            visible_node_mask=visible_mask,
        ),
        visible_node_mask=visible_mask,
    )
    _require_successful_audit(
        audit_transductive_splits(data, splits),
        protocol_name="transductive",
        seed=seed,
    )
    return splits


def generate_comparable_splits(
    data, query=None, test_ratio=None, train_ratio=0.8, *, query_size=None,
    query_fraction=None, validation_query=None, validation_query_size=None,
    validation_query_fraction=None, seed=None, **kwargs
):
    """Generate training, semi-inductive validation, and paired test splits.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Simple undirected source graph without self-loops or duplicate edges.
    query : int or sequence of int or torch.Tensor, optional
        Explicit test query-node IDs.
    test_ratio : float, optional
        Deprecated alias for ``query_fraction``.
    train_ratio : float, optional
        Controls the visible-edge validation holdout: ``1 - train_ratio`` is
        reserved for validation before matching the transductive test size.
    query_size : int, optional
        Exact number of automatically selected query nodes.
    query_fraction : float, optional
        Fraction of eligible nodes automatically selected as queries.
    validation_query : int or sequence of int or torch.Tensor, optional
        Explicit validation query-node IDs.
    validation_query_size : int, optional
        Number of automatically selected validation queries. Defaults to the
        resolved test-query count.
    validation_query_fraction : float, optional
        Fraction of eligible nodes selected as validation queries.
    seed : int, optional
        Random seed used for query selection and edge sampling.
    **kwargs
        Accepted for backward compatibility and ignored.
        
    Returns
    -------
    SplitBundle
        Training, validation, and paired test partitions with protocol metadata.
    """    

    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError("Split for heterogeneous data is not implemented yet.")
    _validate_simple_undirected_graph(data)

    query_nodes = _resolve_query_node_sets(
        data,
        query,
        validation_query,
        query_size=query_size,
        query_fraction=query_fraction,
        validation_query_size=validation_query_size,
        validation_query_fraction=validation_query_fraction,
        seed=seed,
        test_ratio=test_ratio,
    )
    visible_node_mask = _visible_node_mask(data, query_nodes)
    source, target = data.edge_index
    visible_edges = data.edge_index[
        :, visible_node_mask[source] & visible_node_mask[target]
    ]
    validation_positive_edges = _query_positive_edges(
        data, query_nodes.validation, visible_node_mask
    )
    test_positive_edges = _query_positive_edges(
        data, query_nodes.test, visible_node_mask
    )

    temp_data = data.clone()
    temp_data.edge_index = visible_edges
    train, _, test_transductive = _apply_random_link_split(
        temp_data,
        num_val=1.0 - train_ratio,
        num_test=test_positive_edges.size(1),
        seed=seed,
    )

    validation = _semi_inductive_split_data(
        data,
        train.edge_index,
        validation_positive_edges,
        tuple(query_nodes.validation) + tuple(query_nodes.test),
        seed=None if seed is None else seed + 1,
    )
    test_inductive = _semi_inductive_split_data(
        data,
        test_transductive.edge_index,
        test_positive_edges,
        tuple(query_nodes.validation) + tuple(query_nodes.test),
        seed=None if seed is None else seed + 2,
    )

    transductive_negative_edges = _sample_seen_negative_edges(
        data,
        visible_node_mask,
        num_samples=test_transductive.edge_label_index.size(1),
        seed=None if seed is None else seed + 3,
    )
    _set_negative_edges(test_transductive, transductive_negative_edges)

    _attach_split_metadata(
        (
            ("train", "train", "transductive", train),
            ("validation", "validation", "semi-inductive", validation),
            ("transductive", "test", "transductive", test_transductive),
            ("inductive", "test", "semi-inductive", test_inductive),
        ),
        query_nodes,
        data.num_nodes,
        data.edge_index.device,
    )

    splits = SplitBundle(
        train=train,
        validation=validation,
        tests={
            "transductive": test_transductive,
            "semi-inductive": test_inductive,
        },
        metadata=_protocol_metadata(
            "matched-semi-inductive",
            seed,
            parameters={
                "train_ratio": train_ratio,
                "negative_sampling_ratio": 1.0,
                "test_query_selection": _query_selection_metadata(
                    query,
                    query_size,
                    query_fraction if query_fraction is not None else test_ratio,
                ),
                "validation_query_selection": _query_selection_metadata(
                    validation_query,
                    validation_query_size,
                    validation_query_fraction,
                    default_mode="match-test",
                ),
            },
            requested_counts={
                "validation_queries": len(query_nodes.validation),
                "test_queries": len(query_nodes.test),
            },
            train=train,
            validation=validation,
            tests={
                "transductive": test_transductive,
                "semi-inductive": test_inductive,
            },
            visible_node_mask=visible_node_mask,
            validation_queries=len(query_nodes.validation),
            test_queries=len(query_nodes.test),
        ),
        validation_query_nodes=query_nodes.validation,
        test_query_nodes=query_nodes.test,
        visible_node_mask=visible_node_mask,
    )
    _require_successful_audit(
        audit_comparable_splits(data, splits),
        protocol_name="matched-semi-inductive",
        seed=seed,
    )
    return splits


def generate_validation_splits(
    data, *, query_size=None, query_fraction=None,
    validation_query_size=None, validation_query_fraction=None,
    train_ratio=0.8, seed=None,
):
    """Generate training and semi-inductive validation splits.

    Test queries and their incident positives are reserved when sizing the
    training graph, but no test split is returned. This prevents validation-
    only search code from accessing test labels.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Simple undirected source graph.
    query_size, validation_query_size : int, optional
        Exact test and validation query counts.
    query_fraction, validation_query_fraction : float, optional
        Fractions of eligible nodes used for the two query sets.
    train_ratio : float, optional
        Controls the validation holdout: ``1 - train_ratio`` of visible edges
        is reserved for validation, and test-sized capacity is additionally
        withheld to match final benchmark training sizes.
    seed : int, optional
        Seed used for query selection, edge splitting, and negatives.

    Returns
    -------
    SplitBundle
        Training and validation partitions without test partitions.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError(
            "Split for heterogeneous data is not implemented yet."
        )
    _validate_simple_undirected_graph(data)
    if not isinstance(train_ratio, numbers.Real) or not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one.")

    query_nodes = select_query_node_sets(
        data,
        query_size=query_size,
        query_fraction=query_fraction,
        validation_query_size=validation_query_size,
        validation_query_fraction=validation_query_fraction,
        seed=seed,
    )
    visible_node_mask = _visible_node_mask(data, query_nodes)
    source, target = data.edge_index
    visible_edges = data.edge_index[
        :, visible_node_mask[source] & visible_node_mask[target]
    ]
    undirected_visible_edges = visible_edges[
        :, visible_edges[0] < visible_edges[1]
    ]
    validation_positive_edges = _query_positive_edges(
        data, query_nodes.validation, visible_node_mask
    )
    reserved_test_edges = _query_positive_edges(
        data, query_nodes.test, visible_node_mask
    ).size(1)

    num_edges = undirected_visible_edges.size(1)
    num_validation_edges = int((1.0 - train_ratio) * num_edges)
    num_train_edges = num_edges - num_validation_edges - reserved_test_edges
    if num_train_edges <= 0:
        raise ValueError("split ratios leave no edges for training")

    devices = []
    if data.edge_index.device.type == "cuda":
        devices = [data.edge_index.device.index]
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(seed)
        permutation = torch.randperm(num_edges, device=data.edge_index.device)
    train_positive_edges = undirected_visible_edges[
        :, permutation[:num_train_edges]
    ]
    train_message_edges = torch.cat(
        [train_positive_edges, train_positive_edges.flip(0)], dim=1
    )

    train = data.clone()
    train.edge_index = train_message_edges
    train.edge_label_index = train_positive_edges
    train.edge_label = torch.ones(
        train_positive_edges.size(1), device=data.edge_index.device
    )
    validation = _semi_inductive_split_data(
        data,
        train_message_edges,
        validation_positive_edges,
        tuple(query_nodes.validation) + tuple(query_nodes.test),
        seed=None if seed is None else seed + 1,
    )
    _attach_split_metadata(
        (
            ("train", "train", "transductive", train),
            ("validation", "validation", "semi-inductive", validation),
        ),
        query_nodes,
        data.num_nodes,
        data.edge_index.device,
    )
    splits = SplitBundle(
        train=train,
        validation=validation,
        tests={},
        metadata=_protocol_metadata(
            "matched-semi-inductive-validation-only",
            seed,
            parameters={
                "train_ratio": train_ratio,
                "negative_sampling_ratio": 1.0,
                "reserve_test_capacity": True,
                "test_query_selection": _query_selection_metadata(
                    None, query_size, query_fraction
                ),
                "validation_query_selection": _query_selection_metadata(
                    None,
                    validation_query_size,
                    validation_query_fraction,
                    default_mode="match-test",
                ),
            },
            requested_counts={
                "validation_queries": len(query_nodes.validation),
                "reserved_test_queries": len(query_nodes.test),
                "reserved_test_positives": reserved_test_edges,
            },
            train=train,
            validation=validation,
            tests={},
            visible_node_mask=visible_node_mask,
            validation_queries=len(query_nodes.validation),
            reserved_test_queries=len(query_nodes.test),
            reserved_test_positives=reserved_test_edges,
        ),
        validation_query_nodes=query_nodes.validation,
        test_query_nodes=query_nodes.test,
        visible_node_mask=visible_node_mask,
    )
    _require_successful_audit(
        audit_validation_splits(data, splits),
        protocol_name="matched-semi-inductive-validation-only",
        seed=seed,
    )
    return splits



def inductive_split(
    data, query=None, test_ratio=None, train_ratio=0.8, *, query_size=None,
    query_fraction=None, validation_query=None, validation_query_size=None,
    validation_query_fraction=None, seed=None, **kwargs
):
    """Generate training and semi-inductive validation and test splits.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Simple undirected source graph without self-loops or duplicate edges.
    query : int or sequence of int or torch.Tensor, optional
        Explicit test query-node IDs.
    test_ratio : float, optional
        Deprecated alias for ``query_fraction``.
    train_ratio : float, optional
        Fraction of remaining edges assigned to training.
    query_size : int, optional
        Exact number of automatically selected query nodes.
    query_fraction : float, optional
        Fraction of eligible nodes automatically selected as queries.
    validation_query : int or sequence of int or torch.Tensor, optional
        Explicit validation query-node IDs.
    validation_query_size : int, optional
        Number of automatically selected validation queries. Defaults to the
        resolved test-query count.
    validation_query_fraction : float, optional
        Fraction of eligible nodes selected as validation queries.
    seed : int, optional
        Random seed used for query selection and edge sampling.
    **kwargs
        Accepted for backward compatibility and ignored.
        
    Returns
    -------
    SplitBundle
        Training, validation, and semi-inductive test partitions.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError("Split for heterogeneous data is not implemented yet.")
    _validate_simple_undirected_graph(data)

    query_nodes = _resolve_query_node_sets(
        data,
        query,
        validation_query,
        query_size=query_size,
        query_fraction=query_fraction,
        validation_query_size=validation_query_size,
        validation_query_fraction=validation_query_fraction,
        seed=seed,
        test_ratio=test_ratio,
    )
    visible_node_mask = _visible_node_mask(data, query_nodes)
    source, target = data.edge_index
    visible_edges = data.edge_index[
        :, visible_node_mask[source] & visible_node_mask[target]
    ]
    validation_positive_edges = _query_positive_edges(
        data, query_nodes.validation, visible_node_mask
    )
    test_positive_edges = _query_positive_edges(
        data, query_nodes.test, visible_node_mask
    )

    temp_data = data.clone()
    temp_data.edge_index = visible_edges
    train, _, _ = _apply_random_link_split(
        temp_data,
        num_val=1.0 - train_ratio,
        num_test=0,
        seed=seed,
    )

    all_queries = tuple(query_nodes.validation) + tuple(query_nodes.test)
    validation = _semi_inductive_split_data(
        data,
        train.edge_index,
        validation_positive_edges,
        all_queries,
        seed=None if seed is None else seed + 1,
    )
    test = _semi_inductive_split_data(
        data,
        visible_edges,
        test_positive_edges,
        all_queries,
        seed=None if seed is None else seed + 2,
    )

    _attach_split_metadata(
        (
            ("train", "train", "transductive", train),
            ("validation", "validation", "semi-inductive", validation),
            ("inductive", "test", "semi-inductive", test),
        ),
        query_nodes,
        data.num_nodes,
        data.edge_index.device,
    )

    splits = SplitBundle(
        train=train,
        validation=validation,
        tests={"semi-inductive": test},
        metadata=_protocol_metadata(
            "semi-inductive",
            seed,
            parameters={
                "train_ratio": train_ratio,
                "negative_sampling_ratio": 1.0,
                "test_query_selection": _query_selection_metadata(
                    query,
                    query_size,
                    query_fraction if query_fraction is not None else test_ratio,
                ),
                "validation_query_selection": _query_selection_metadata(
                    validation_query,
                    validation_query_size,
                    validation_query_fraction,
                    default_mode="match-test",
                ),
            },
            requested_counts={
                "validation_queries": len(query_nodes.validation),
                "test_queries": len(query_nodes.test),
            },
            train=train,
            validation=validation,
            tests={"semi-inductive": test},
            visible_node_mask=visible_node_mask,
            validation_queries=len(query_nodes.validation),
            test_queries=len(query_nodes.test),
        ),
        validation_query_nodes=query_nodes.validation,
        test_query_nodes=query_nodes.test,
        visible_node_mask=visible_node_mask,
    )
    _require_successful_audit(
        audit_semi_inductive_splits(data, splits),
        protocol_name="semi-inductive",
        seed=seed,
    )
    return splits

def calculate_reduced_features(
    data, dim=128, reduction_method="pca", fit_nodes=None, random_state=None,
    **kwargs
):
    """Fit a reducer and attach reduced node features to a graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph containing node features in ``x``.
    dim : int, optional
        Number of output dimensions.
    reduction_method : str, optional
        Registered reduction method.
    fit_nodes : sequence or torch.Tensor, optional
        Node indices or mask used to fit the reducer.
    random_state : int, optional
        Random state passed to the reducer.
    **kwargs
        Additional reducer-constructor arguments.

    Returns
    -------
    torch_geometric.data.Data
        The input graph with ``x_reduced`` and its fitted reducer attached.
    """

    reducer = fit_reducer(
        data.x,
        fit_nodes=fit_nodes,
        method=reduction_method,
        n_components=dim,
        random_state=random_state,
        **kwargs,
    )
    return attach_reduced_features(data, reducer)

def update_reduced_features(data, reduction_model = None, **kwargs):
    """Update reduced node features using a fitted reducer.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph containing node features in ``x``.
    reduction_model : object, optional
        Fitted reducer. Defaults to ``data.feature_reducer`` and then the
        legacy ``data.dim_reduction`` alias.
    **kwargs
        Accepted for backward compatibility and ignored.

    Returns
    -------
    torch_geometric.data.Data
        The input graph with refreshed ``x_reduced`` values.
    """
    if reduction_model is None:
        reduction_model = getattr(
            data, "feature_reducer", getattr(data, "dim_reduction", None)
        )
    if reduction_model is None:
        raise ValueError("data does not contain a fitted feature reducer")
    return attach_reduced_features(data, reduction_model)
