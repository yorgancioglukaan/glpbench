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
    def __init__(self, train_data = None, **kwargs):
        
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


    
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=self.train_data.edge_index, num_nodes=self.train_data.num_nodes,
            num_neg_samples=self.train_data.edge_label_index.size(1))

        edge_label_index = torch.cat(
            [self.train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            self.train_data.edge_label,
            self.train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = self.model.decode(z, edge_label_index, self.train_data.x_reduced).view(-1)
        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        out = self.model.decode(z, data.edge_label_index, data.x_reduced).view(-1).sigmoid()
        return out