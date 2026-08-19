from torch_geometric.nn import GCNConv, GraphConv, SAGEConv
import torch

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
