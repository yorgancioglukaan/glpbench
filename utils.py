import math
import numbers
import random
import warnings

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import is_undirected, negative_sampling

from feature_reduction import attach_reduced_features, fit_reducer


def _attach_split_metadata(splits, query, num_nodes, device):
    """Attach query membership and semantic names to graph splits."""
    query_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    query_mask[query] = True
    for split_type, split in splits:
        split.query_mask = query_mask.clone()
        split.seen_node_mask = ~query_mask
        split.split_type = split_type


def _node_neighbors(data):
    """Return non-self neighbors for every node."""
    neighbors = [set() for _ in range(data.num_nodes)]
    source, target = data.edge_index.detach().cpu().tolist()
    for src, dst in zip(source, target):
        if src != dst:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return neighbors


def _validate_query_nodes(data, query):
    """Normalize and validate an explicitly supplied query set."""
    if isinstance(query, bool):
        raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
    if isinstance(query, int):
        processed_query = [query]
    elif isinstance(query, list):
        if not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in query
        ):
            raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
        processed_query = query
    elif isinstance(query, torch.Tensor):
        if query.ndim != 1:
            raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
        if query.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise TypeError("Query tensor must use an integer dtype.")
        processed_query = query.detach().cpu().tolist()
    else:
        raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")

    if not processed_query:
        raise ValueError("Query must contain at least one node ID.")
    if any(node < 0 for node in processed_query):
        raise ValueError("Query node IDs must be non-negative integers.")
    if any(node >= data.num_nodes for node in processed_query):
        raise ValueError("Query includes node IDs outside the graph.")
    if len(set(processed_query)) != len(processed_query):
        raise ValueError("Query contains duplicate node IDs.")

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


def _sample_inductive_negative_edges(data, positive_edges, query):
    """Sample non-edges targeting the positive edges' query nodes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Original graph.
    positive_edges : torch.Tensor
        Positive edges with shape ``[2, num_edges]``.
    query : sequence of int
        Query node IDs excluded from candidate sources.

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

    negative_sources = []
    negative_targets = positive_edges[1].detach().cpu().tolist()
    for target in negative_targets:
        candidates = [
            node for node in range(data.num_nodes)
            if node not in query_nodes
            and node != target
            and node not in neighbors[target]
        ]
        if not candidates:
            raise ValueError(
                f"Cannot sample a negative edge for query node {target}: "
                "no eligible non-neighbor nodes are available."
            )
        negative_sources.append(random.choice(candidates))

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


def generate_comparable_splits(
    data, query=None, test_ratio=None, train_ratio=0.8, *, query_size=None,
    query_fraction=None, seed=None, **kwargs
):
    """Generate comparable inductive and transductive splits.

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
    seed : int, optional
        Random seed used for automatic query selection.
        
    Returns
    -------
    tuple of torch_geometric.data.Data
        Training, validation, transductive-test, and inductive-test splits.
    """    

#check if parameters are valid

    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError("Split for heterogeneous data is not implemented yet.")
    if not is_undirected(data.edge_index):
        raise NotImplementedError("Split for directed networks is not implemented yet.")

    query = _resolve_query_nodes(
        data,
        query,
        query_size=query_size,
        query_fraction=query_fraction,
        seed=seed,
        test_ratio=test_ratio,
    )
    query_tensor = torch.tensor(
        query, dtype=data.edge_index.dtype, device=data.edge_index.device
    )

    #get edges incident on queries
   
    src = data.edge_index[0,:]
    tgt = data.edge_index[1,:]
    src_mask = torch.isin(src, query_tensor)
    tgt_mask = torch.isin(tgt, query_tensor)
    mask = src_mask | tgt_mask
    test_edges = data.edge_index[:,mask]
    other_edges = data.edge_index[:, ~mask]

    test_inductive = data.clone()
    directed_test_edges = test_edges[:, ~torch.isin(test_edges[0, :], query_tensor)]
    test_inductive.edge_index = other_edges
    test_inductive.edge_label_index = directed_test_edges
    n_test_edges = directed_test_edges.size()[1]
    test_inductive.edge_label = torch.ones(n_test_edges)

    #negative sampling for test edges

    negative_edges = _sample_inductive_negative_edges(data, directed_test_edges, query)
    negative_labels = torch.zeros(n_test_edges)
    test_inductive.edge_label = torch.cat([test_inductive.edge_label, negative_labels],dim=0)
    test_inductive.edge_label_index = torch.cat([test_inductive.edge_label_index, negative_edges], dim=1)

    # test_transductive, training and validation splits

    temp_data = data.clone()
    temp_data.edge_index = other_edges

    link_splitter = T.RandomLinkSplit(
        num_val=1.0 - train_ratio,
        num_test=n_test_edges,
        add_negative_train_samples=False,
        is_undirected=True,
        neg_sampling_ratio=0.0,
    )
    
    train, val, test_transductive = link_splitter(temp_data)
    seen_node_mask = torch.ones(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    seen_node_mask[query] = False
    validation_negative_edges = _sample_seen_negative_edges(
        data,
        seen_node_mask,
        num_samples=val.edge_label_index.size(1),
        seed=seed,
    )
    _set_negative_edges(val, validation_negative_edges)

    transductive_negative_edges = _sample_seen_negative_edges(
        data,
        seen_node_mask,
        num_samples=test_transductive.edge_label_index.size(1),
        seed=None if seed is None else seed + 1,
    )
    _set_negative_edges(test_transductive, transductive_negative_edges)
    test_inductive.edge_index = test_transductive.edge_index

    _attach_split_metadata(
        (
            ("train", train),
            ("validation", val),
            ("transductive", test_transductive),
            ("inductive", test_inductive),
        ),
        query,
        data.num_nodes,
        data.edge_index.device,
    )

    return train, val, test_transductive, test_inductive



def inductive_split(
    data, query=None, test_ratio=None, train_ratio=0.8, *, query_size=None,
    query_fraction=None, seed=None, **kwargs
):
    """Hold out edges incident to cold-start query nodes.

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
    seed : int, optional
        Random seed used for automatic query selection.
        
    Returns
    -------
    tuple of torch_geometric.data.Data
        Training, validation, and inductive-test splits.
    """

    #check if parameters are valid

    if not isinstance(data, Data):
        raise TypeError("data should be of type torch_geometric.data.Data")
    if isinstance(data, HeteroData):
        raise NotImplementedError("Split for heterogeneous data is not implemented yet.")
    if not is_undirected(data.edge_index):
        raise NotImplementedError("Split for directed networks is not implemented yet.")

    query = _resolve_query_nodes(
        data,
        query,
        query_size=query_size,
        query_fraction=query_fraction,
        seed=seed,
        test_ratio=test_ratio,
    )
    query_tensor = torch.tensor(
        query, dtype=data.edge_index.dtype, device=data.edge_index.device
    )


    #get edges incident on queries
   
    src = data.edge_index[0,:]
    tgt = data.edge_index[1,:]
    src_mask = torch.isin(src, query_tensor)
    tgt_mask = torch.isin(tgt, query_tensor)
    mask = src_mask | tgt_mask
    test_edges = data.edge_index[:,mask]
    other_edges = data.edge_index[:, ~mask]

    # split train/val edges
    link_splitter = T.RandomLinkSplit(
        num_val=1.0 - train_ratio,
        num_test=0,
        add_negative_train_samples=False,
        is_undirected=True,
        neg_sampling_ratio=0.0,
    )
    temp_data = data.clone()
    test = data.clone()
    temp_data.edge_index = other_edges


    train, val, _ = link_splitter(temp_data)

    seen_node_mask = torch.ones(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    seen_node_mask[query] = False
    validation_negative_edges = _sample_seen_negative_edges(
        data,
        seen_node_mask,
        num_samples=val.edge_label_index.size(1),
        seed=seed,
    )
    _set_negative_edges(val, validation_negative_edges)

    directed_test_edges = test_edges[:, ~torch.isin(test_edges[0, :], query_tensor)]
    test.edge_index = other_edges
    test.edge_label_index = directed_test_edges
    n_test_edges = directed_test_edges.size()[1]
    test.edge_label = torch.ones(n_test_edges)

    #negative sampling for test edges

    negative_edges = _sample_inductive_negative_edges(data, directed_test_edges, query)
    negative_labels = torch.zeros(n_test_edges)
    test.edge_label = torch.cat([test.edge_label, negative_labels],dim=0)
    test.edge_label_index = torch.cat([test.edge_label_index, negative_edges], dim=1)

    _attach_split_metadata(
        (("train", train), ("validation", val), ("inductive", test)),
        query,
        data.num_nodes,
        data.edge_index.device,
    )

    return train, val, test

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
