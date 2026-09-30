import math
import numbers
import random
import warnings
from typing import NamedTuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import is_undirected, negative_sampling

from feature_reduction import attach_reduced_features, fit_reducer


class QueryNodeSets(NamedTuple):
    """Store disjoint validation and test query-node IDs."""

    validation: tuple[int, ...]
    test: tuple[int, ...]


class ComparableSplits(NamedTuple):
    """Store training, validation, and paired test graph splits."""

    train: Data
    validation: Data
    test_transductive: Data
    test_inductive: Data


class InductiveSplits(NamedTuple):
    """Store training, validation, and inductive-test graph splits."""

    train: Data
    validation: Data
    test_inductive: Data


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
    for split_type, split_stage, split_setting, split in splits:
        split.validation_query_mask = validation_query_mask.clone()
        split.test_query_mask = test_query_mask.clone()
        split.visible_node_mask = visible_node_mask.clone()

        # Compatibility aliases retained while callers migrate to node roles.
        split.query_mask = query_mask.clone()
        split.seen_node_mask = visible_node_mask.clone()
        split.split_type = split_type

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


def select_query_node_sets(
    data, *, test_query=None, validation_query=None, query_size=None,
    query_fraction=None, validation_query_size=None,
    validation_query_fraction=None, seed=None, max_attempts=1000,
):
    """Select disjoint validation and test query-node sets.

    Every selected query retains at least one neighbor outside the union of
    both query sets. If validation sizing is omitted, its node count matches
    the resolved test-query count.
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
    for _ in range(max_attempts):
        available = [
            node for node in candidates
            if node not in fixed_test_set and node not in fixed_validation_set
        ]
        missing_test = test_count - len(fixed_test_set)
        missing_validation = validation_count - len(fixed_validation_set)
        if missing_test + missing_validation > len(available):
            break
        sampled = rng.sample(available, missing_test + missing_validation)
        test_nodes = sorted(fixed_test_set | set(sampled[:missing_test]))
        validation_nodes = sorted(
            fixed_validation_set | set(sampled[missing_test:])
        )
        query_union = set(test_nodes) | set(validation_nodes)
        if all(neighbors[node].difference(query_union) for node in query_union):
            return QueryNodeSets(
                validation=tuple(validation_nodes), test=tuple(test_nodes)
            )

    raise ValueError(
        "Could not select disjoint validation and test query nodes for which "
        f"every query retains a training-visible neighbor after {max_attempts} "
        "attempts."
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
    for _ in range(max_attempts):
        selected = sorted(rng.sample(candidates, resolved_size))
        selected_set = set(selected)
        if all(neighbors[node].difference(selected_set) for node in selected):
            return selected

    raise ValueError(
        f"Could not select {resolved_size} query nodes for which every query "
        f"retains a visible neighbor after {max_attempts} attempts."
    )


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


def audit_comparable_splits(data, splits):
    """Raise ``ValueError`` if a comparable split violates leakage rules.

    The audit covers node roles, message-passing topology, positive-set
    disjointness, negative-edge validity, endpoint orientation, and matching
    of the paired test sets.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if not isinstance(splits, ComparableSplits):
        raise TypeError("splits should be of type ComparableSplits")
    _validate_simple_undirected_graph(data)

    split_items = tuple(zip(ComparableSplits._fields, splits))
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
        splits.test_transductive.edge_index, splits.test_inductive.edge_index
    ):
        raise ValueError("The paired test message-passing graphs differ.")
    if (
        splits.test_transductive.edge_label_index.size(1)
        != splits.test_inductive.edge_label_index.size(1)
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
        ("test_inductive", splits.test_inductive, test_mask),
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

    transductive_nodes = splits.test_transductive.edge_label_index
    if transductive_nodes.numel() and not bool(visible_mask[transductive_nodes].all()):
        raise ValueError("Transductive labeled edges contain a query node.")
    if int((splits.test_transductive.edge_label == 1).sum()) != int(
        (splits.test_transductive.edge_label == 0).sum()
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

    return None


def generate_comparable_splits(
    data, query=None, test_ratio=None, train_ratio=0.8, *, query_size=None,
    query_fraction=None, validation_query=None, validation_query_size=None,
    validation_query_fraction=None, seed=None, **kwargs
):
    """Generate training, semi-inductive validation, and paired test splits.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Undirected source graph.
    query : int or sequence of int or torch.Tensor, optional
        Cold-start node IDs.
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
        
    Returns
    -------
    ComparableSplits
        Named training, validation, transductive-test, and inductive-test splits.
        The result also supports tuple unpacking for backward compatibility.
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

    splits = ComparableSplits(
        train, validation, test_transductive, test_inductive
    )
    audit_comparable_splits(data, splits)
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
        Undirected source graph.
    query : int or sequence of int or torch.Tensor, optional
        Cold-start node IDs.
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
        
    Returns
    -------
    InductiveSplits
        Named training, validation, and inductive-test splits. The result also
        supports tuple unpacking for backward compatibility.
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

    return InductiveSplits(train, validation, test)

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
        Reducer to use. Defaults to ``data.dim_reduction``.
    """
    if reduction_model is None:
        reduction_model = getattr(
            data, "feature_reducer", getattr(data, "dim_reduction", None)
        )
    if reduction_model is None:
        raise ValueError("data does not contain a fitted feature reducer")
    return attach_reduced_features(data, reduction_model)
