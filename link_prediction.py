from torch_geometric.nn import GCNConv, GraphConv, SAGEConv
import torch
from torch_geometric.utils import negative_sampling

class AGAE_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv_layer_type = GCNConv, **kwargs):
        super().__init__()

        #create convolution layers

        if conv_layer_type == 'GCNConv':
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_layer_type == 'GraphConv':
                self.conv1 = GraphConv(in_channels, hidden_channels)
                self.conv2 = GraphConv(hidden_channels, out_channels)
        elif conv_layer_type == 'SAGEConv':
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
             raise ValueError("conv_layer_type must be one of 'GCNConv', 'GraphConv', 'SAGEConv'")           

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index, x_reduced):
        return (z[edge_label_index[0]] * x_reduced[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, x_reduced):
        prob_adj = z @ x_reduced.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()

class GAE_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv_layer_type = GCNConv, **kwargs):
        super().__init__()

        #create convolution layers

        if conv_layer_type == 'GCNConv':
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_layer_type == 'GraphConv':
                self.conv1 = GraphConv(in_channels, hidden_channels)
                self.conv2 = GraphConv(hidden_channels, out_channels)
        elif conv_layer_type == 'SAGEConv':
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
             raise ValueError("conv_layer_type must be one of 'GCNConv', 'GraphConv', 'SAGEConv'")

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()

class LinkPrediction:
    '''
    Default Link Prediction module for all types of graph auto encoder variants

    ...

    Attributes:
    -----------
    model_type : str
        symmetric, asymmetric, and other variants
    optimizer : torch.optim.Optimizer
        torch.optim.Adam by default
    criterion : torch.nn.ModuleHolder
        toch.nn.BCEWithLogitsLoss by default
    model : torch.nn.Module
        underlying graph auto-encoder model

    
    '''

    def __init__(self, train_data, **kwargs):
        '''
        Constructor for LinkPrediction object

        Parameters:
        -----------
        train_data : torch_geometric.data.Data
            Training data to be used for back-propagation
        
        '''

        
        self.model_type = kwargs["model_type"] if "model_type" in kwargs.keys() else "symmetric"

        hidden_channels = kwargs["hidden_channels"] if "in_channels" in kwargs.keys() else 128
        out_channels = kwargs["out_channels"] if "out_channels" in kwargs.keys() else 64
        conv_layer_type= kwargs["conv_layer_type"] if "conv_layer_type" in kwargs.keys() else None

        self.model = None
        if self.model_type == "symmetric":
             self.model = GAE_Model(in_channels=train_data.num_features, 
                                    hidden_channels=hidden_channels,
                                    out_channels=out_channels,
                                    conv_layer_type= conv_layer_type,
                                    kwargs = kwargs)
        elif self.model_type == "asymmetric":
             self.model = AGAE_Model(in_channels=train_data.num_features, 
                                    hidden_channels=hidden_channels,
                                    out_channels=out_channels,
                                    conv_layer_type= conv_layer_type,
                                    kwargs = kwargs)
        else:
             raise ValueError("Unexpected model type: " + self.model_type)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, train_data = None):
        '''
        Trains the model for one epoch by performing one round of back-propagation using `self.training_data`

        Parameters:
        -----------
        train_data : torch_geometric.data.Data, optional
            If supplied with a training data, the method uses it for back propagation. 
            Otherwise it uses `self.train_data` provided during construction.

        Returns:
        --------
        loss : float
            output of the loss function calculated before back-propagation

        '''

        if train_data == None:
             train_data = self.train_data


        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1))

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = self.model.decode(z, edge_label_index, train_data.x_reduced).view(-1)
        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        '''
        Uses trained model to predict for the provided test data

        Parameters:
        -----------
        data : torch_geometric.data.Data
            test data to make predictions for. Assumes edge_label and edge_label_index are present.

        '''
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        out = self.model.decode(z, data.edge_label_index, data.x_reduced).view(-1).sigmoid()
        return out