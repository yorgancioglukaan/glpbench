import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import is_undirected
import random
from feature_reduction import attach_reduced_features, fit_reducer


def _attach_split_metadata(splits, query, num_nodes, device):
    """Attach query membership and semantic names to graph splits."""
    query_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    query_mask[query] = True
    for split_type, split in splits:
        split.query_mask = query_mask.clone()
        split.seen_node_mask = ~query_mask
        split.split_type = split_type


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


def generate_comparable_splits(data, query=None, test_ratio=0.1, train_ratio=0.8, **kwargs):
    """Generate comparable inductive and transductive splits.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Undirected source graph.
    query : int or sequence of int or torch.Tensor, optional
        Cold-start node IDs.
    test_ratio : float, optional
        Fraction used for automatic query selection.
    train_ratio : float, optional
        Fraction of remaining edges assigned to training.
        
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

    #handle query
    processed_query = None
    if isinstance(query, int):
        if query < 0: 
            raise ValueError("Encountered negative number(s) in query. All node IDs in query must be positive integers.")
        else:
            processed_query = [query]
    elif isinstance(query, list):
        if any(isinstance(item, list) for item in query):
            raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
        elif any((item < 0) for item in query):
            raise ValueError("Encountered negative number(s) in query. All node IDs in query must be positive integers.")
        elif not all(isinstance(item, int) for item in query):
            raise TypeError("Query is of type list but contains non-int elements.")
        else:
            processed_query = query
    elif isinstance(query, torch.Tensor):
        if query.ndim != 1:
            raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
        elif not query.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("If query is a tensor it must have a dtype of either int8, int16, int32m int64, or unint8.")
        elif any((item < 0) for item in query):
            raise ValueError("Encountered negative number(s) in query. All node IDs in query must be positive integers.")
        else:
            processed_query = query.tolist()
    elif query == None:
        #TODO
        raise NotImplementedError("Automatic query generation not yet implemented")
    else:
        raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")

    query = processed_query
    if not all((item < data.num_nodes) for item in query):
        raise ValueError("Query includes node IDs larger than number of nodes in data.")

    #get edges incident on queries
   
    src = data.edge_index[0,:]
    tgt = data.edge_index[1,:]
    src_mask = torch.isin(src, torch.tensor(query))
    tgt_mask = torch.isin(tgt, torch.tensor(query))
    mask = src_mask | tgt_mask
    test_edges = data.edge_index[:,mask]
    other_edges = data.edge_index[:, ~mask]

    test_inductive = data.clone()
    directed_test_edges = test_edges[:,~torch.isin(test_edges[0,:], torch.tensor(query))]
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

    link_splitter = T.RandomLinkSplit(num_val=1.0-train_ratio, num_test=n_test_edges, 
                                      add_negative_train_samples=False, is_undirected=True,
                                      neg_sampling_ratio=1.0)
    
    train, val, test_transductive = link_splitter(temp_data)
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



def inductive_split(data, query=None, test_ratio=0.1, train_ratio=0.8, **kwargs):
    """Hold out edges incident to cold-start query nodes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Undirected source graph.
    query : int or sequence of int or torch.Tensor, optional
        Cold-start node IDs.
    test_ratio : float, optional
        Fraction used for automatic query selection.
    train_ratio : float, optional
        Fraction of remaining edges assigned to training.
        
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

    #handle query
    processed_query = None
    if isinstance(query, int):
        if query < 0: 
            raise ValueError("Encountered negative number(s) in query. All node IDs in query must be positive integers.")
        else:
            processed_query = [query]
    elif isinstance(query, list):
        if any(isinstance(item, list) for item in query):
            raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
        elif any((item < 0) for item in query):
            raise ValueError("Encountered negative number(s) in query. All node IDs in query must be positive integers.")
        elif not all(isinstance(item, int) for item in query):
            raise TypeError("Query is of type list but contains non-int elements.")
        else:
            processed_query = query
    elif isinstance(query, torch.Tensor):
        if query.ndim != 1:
            raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")
        elif not query.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("If query is a tensor it must have a dtype of either int8, int16, int32m int64, or unint8.")
        elif any((item < 0) for item in query):
            raise ValueError("Encountered negative number(s) in query. All node IDs in query must be positive integers.")
        else:
            processed_query = query.tolist()
    elif query == None:
        #TODO
        raise NotImplementedError("Automatic query generation not yet implemented")
    else:
        raise TypeError("Query must be an int or a 1D list or 1D torch tensor.")

    query = processed_query
    if not all((item < data.num_nodes) for item in query):
        raise ValueError("Query includes node IDs larger than number of nodes in data.")


    #get edges incident on queries
   
    src = data.edge_index[0,:]
    tgt = data.edge_index[1,:]
    src_mask = torch.isin(src, torch.tensor(query))
    tgt_mask = torch.isin(tgt, torch.tensor(query))
    mask = src_mask | tgt_mask
    test_edges = data.edge_index[:,mask]
    other_edges = data.edge_index[:, ~mask]

    # split train/val edges
    link_splitter = T.RandomLinkSplit(num_val=1.0-train_ratio, num_test=0, 
                                      add_negative_train_samples=False, is_undirected=True)
    temp_data = data.clone()
    test = data.clone()
    temp_data.edge_index = other_edges


    train, val, _ = link_splitter(temp_data)

    directed_test_edges = test_edges[:,~torch.isin(test_edges[0,:], torch.tensor(query))]
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
