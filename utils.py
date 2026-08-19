import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import is_undirected
import random



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

    #randomly sample candidate nodes with replacement
    cand_nodes = list(set(range(data.num_nodes-1))- set(query))
    negative_src = torch.tensor(random.choices(cand_nodes,k=n_test_edges))
    negative_tgt = directed_test_edges[1,:]
    negative_edges = torch.stack([negative_src,negative_tgt], dim=0)
    negative_labels = torch.zeros(n_test_edges)
    test.edge_label = torch.cat([test.edge_label, negative_labels],dim=0)
    test.edge_label_index = torch.cat([test.edge_label_index, negative_edges], dim=1)

    return train, val, test