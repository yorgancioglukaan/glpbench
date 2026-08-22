import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import is_undirected
import random
from sklearn.decomposition import PCA


def _sample_inductive_negative_edges(data, positive_edges, query):
    """Sample non-edges ending at the same query nodes as ``positive_edges``."""
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
    """Generate both inductive and trandcutive split for a dataset. 
    
    This function takes in a PyG Data and outputs train, val and test sets for both inductive and transductive settings. 

    Parameters
    ----------
    data : torch_geometric.data.Data
        Source network
    query : int or list or torch.Tensor, optional
        Node id(s) of cold-start node(s). Must be an int or a list of integeres or a 1 dimesnional tensor of integers.
    test_ratio : float, optional
        If a query is not provided, uses test ratio to select a number of query nodes.
    train_ratio : float, optional
        After test sets are sampled, uses this ratio to calculate training and validation sets.
        
    Returns
    -------
    train : torch_geometric.data.Data
        Training network.
    val : torch_geometric.data.Data
        Validation network.
    test_transductive : torch_geometric.data.Data
        Transductive testing network. `edge_label_index` is generated to be the same size as the inductive_split's
    test_inductive : torch_geometric.data.Data
        Inductive testing network. `edge_label_index`ontains only the edges incident on query. Has the same `edge_index` as transductive split.
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

    return train, val, test_transductive, test_inductive



def inductive_split(data, query=None, test_ratio=0.1, train_ratio=0.8, **kwargs):
    """Perform inductive split for a dataset. AKA cold-start, node-disjoint, unseen-node, held-out-node.
    
    This function takes in a PyG Data and outputs train, val and test inductive splits. Split is "train rest" unless specified otherwise. 
    Split is generated such that any edges incident on query node(s) are reserved for the test set. Remaining Edges are split between training and validation.
    If a query is not specified, it is generated automatically using the test_ratio.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Source network
    query : int or list or torch.Tensor, optional
        Node id(s) of cold-start node(s). Must be an int or a list of integeres or a 1 dimesnional tensor of integers.
    
        
    Returns
    -------
    train : torch_geometric.data.Data
        Training network.
    val : torch_geometric.data.Data
        Validation network.
    test : torch_geometric.data.Data
        Testing network. Contains only the edges incident on query.

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

    return train, val, test

def calculate_reduced_features(data, dim = 128, reduction_method = 'pca', **kwargs):
    '''
    This function adds an attribute `data.x_reduced` do the provided data object. Where `x_reduced` is the dimension-reduced feature matrix.
    It also saves the calculated dimensionality reduction model to `data.dim_reduction` that can be used for future applications to new data.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        Source network to be operated on. 
    dim : int, optional
        Number of dimensions to reduce the features to. Default is 128.
    reduction_method : str, optional
        Reduction model to be used. Must be one of 'pca'. Default is pca
    '''

    valid_reduction_methods = ['pca']
    if not reduction_method in valid_reduction_methods:
        raise ValueError(f"reduction method should be one of {', '.join(valid_reduction_methods)}")

    if reduction_method == 'pca':
        data.dim_reduction = PCA(n_components=dim)
        data.dim_reduction.fit(data.x)
        data.x_reduced = torch.tensor(data.dim_reduction.fit_transform(data.x.cpu()), dtype=torch.float).to(data.x.device)

def update_reduced_features(data, reduction_model = None, **kwargs):
    '''
    This function appliess the previously trained dimensionality reduction method to new data
    
    Parameters:
    -----------
    data : torch_geometric.data.Data
        Network to be operated on.
    reduction_model : object, optional
        If provided, the function will use this model instead of the one that is saved on the `data.dim_reduction`
    '''
    if reduction_model == None:
        reduction_model = data.dim_reduction

    if isinstance(reduction_model, PCA):
        data.x_reduced = torch.tensor(reduction_model.fit_transform(data.x.cpu()), dtype=torch.float).to(data.x.device)
