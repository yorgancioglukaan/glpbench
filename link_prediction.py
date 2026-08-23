from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, SAGEConv
import torch
from torch_geometric.utils import negative_sampling


_GATV2_ARGUMENTS = {
    "heads",
    "concat",
    "negative_slope",
    "dropout",
    "add_self_loops",
    "edge_dim",
    "fill_value",
    "bias",
    "share_weights",
}


def _gatv2_layers(in_channels, hidden_channels, out_channels, kwargs):
    """Construct dimensionally compatible GATv2 encoder layers."""
    gatv2_kwargs = kwargs.get("gatv2_kwargs", {})
    if not isinstance(gatv2_kwargs, dict):
        raise TypeError("gatv2_kwargs must be a dictionary")
    gatv2_kwargs = gatv2_kwargs.copy()
    gatv2_kwargs.update({
        name: kwargs[name] for name in _GATV2_ARGUMENTS if name in kwargs
    })

    heads = gatv2_kwargs.get("heads", 1)
    concat = gatv2_kwargs.get("concat", True)
    conv1 = GATv2Conv(in_channels, hidden_channels, **gatv2_kwargs)
    conv1_out_channels = hidden_channels * heads if concat else hidden_channels

    output_kwargs = gatv2_kwargs.copy()
    output_kwargs.update({"heads": 1, "concat": False})
    conv2 = GATv2Conv(conv1_out_channels, out_channels, **output_kwargs)
    return conv1, conv2


class AGAE_Model(torch.nn.Module):
    """Asymmetric graph autoencoder.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden feature dimension.
    out_channels : int
        Latent feature dimension.
    conv_layer_type : str, optional
        Message-passing layer type. GATv2 options may be passed through
        ``gatv2_kwargs`` or as keyword arguments.
    """

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
        elif conv_layer_type == 'GATv2Conv':
                self.conv1, self.conv2 = _gatv2_layers(
                    in_channels, hidden_channels, out_channels, kwargs
                )
        else:
             raise ValueError("conv_layer_type must be one of 'GCNConv', 'GraphConv', 'SAGEConv', 'GATv2Conv'")

    def encode(self, x, edge_index):
        """Encode nodes into latent representations."""
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index, x_reduced):
        """Score edges using latent sources and feature-based targets."""
        return (z[edge_label_index[0]] * x_reduced[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z, x_reduced):
        """Return node pairs with reconstruction scores above 0.5."""
        prob_adj = z @ x_reduced.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()

class GAE_Model(torch.nn.Module):
    """Symmetric graph autoencoder.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden feature dimension.
    out_channels : int
        Latent feature dimension.
    conv_layer_type : str, optional
        Message-passing layer type. GATv2 options may be passed through
        ``gatv2_kwargs`` or as keyword arguments.
    """

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
        elif conv_layer_type == 'GATv2Conv':
                self.conv1, self.conv2 = _gatv2_layers(
                    in_channels, hidden_channels, out_channels, kwargs
                )
        else:
             raise ValueError("conv_layer_type must be one of 'GCNConv', 'GraphConv', 'SAGEConv', 'GATv2Conv'")

    def encode(self, x, edge_index):
        """Encode nodes into latent representations."""
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """Score edges using pairwise latent dot products."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """Return node pairs with reconstruction scores above 0.5."""
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()

class LinkPrediction:
    """Train and evaluate graph-autoencoder link predictors.

    Parameters
    ----------
    train_data : torch_geometric.data.Data
        Graph used for training.

    Attributes
    ----------
    model_type : str
        Symmetric or asymmetric architecture.
    optimizer : torch.optim.Optimizer
        Model optimizer.
    criterion : torch.nn.Module
        Binary classification loss.
    model : torch.nn.Module
        Underlying graph autoencoder.
    """

    def __init__(self, train_data, **kwargs):
        """Initialize a link-prediction trainer.

        Parameters
        ----------
        train_data : torch_geometric.data.Data
            Graph used for training.
        **kwargs
            Model type, layer type, and channel dimensions.
        """

        
        self.train_data = train_data
        self.model_type = kwargs["model_type"] if "model_type" in kwargs.keys() else "symmetric"

        hidden_channels = kwargs["hidden_channels"] if "hidden_channels" in kwargs.keys() else 128
        out_channels = kwargs["out_channels"] if "out_channels" in kwargs.keys() else 64
        conv_layer_type= kwargs["conv_layer_type"] if "conv_layer_type" in kwargs.keys() else "GCNConv"
        model_kwargs = {
            name: value for name, value in kwargs.items()
            if name not in {
                "model_type", "hidden_channels", "out_channels", "conv_layer_type"
            }
        }

        self.model = None
        if self.model_type == "symmetric":
             self.model = GAE_Model(in_channels=train_data.num_features, 
                                    hidden_channels=hidden_channels,
                                    out_channels=out_channels,
                                    conv_layer_type= conv_layer_type,
                                    **model_kwargs)
        elif self.model_type == "asymmetric":
             self.model = AGAE_Model(in_channels=train_data.num_features, 
                                    hidden_channels=hidden_channels,
                                    out_channels=out_channels,
                                    conv_layer_type= conv_layer_type,
                                    **model_kwargs)
        else:
             raise ValueError("Unexpected model type: " + self.model_type)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, train_data = None):
        """Train the model for one epoch.

        Parameters
        ----------
        train_data : torch_geometric.data.Data, optional
            Training graph. Defaults to the graph supplied at construction.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """

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

        if self.model_type == "asymmetric":
            out = self.model.decode(z, edge_label_index, train_data.x_reduced).view(-1)
        else:
            out = self.model.decode(z, edge_label_index).view(-1)
        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def test(self, data):
        """Predict edge probabilities for a labeled graph split.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph containing ``edge_label_index``.

        Returns
        -------
        torch.Tensor
            Predicted edge probabilities.
        """
        self.model.eval()
        z = self.model.encode(data.x, data.edge_index)
        if self.model_type == "asymmetric":
            out = self.model.decode(z, data.edge_label_index, data.x_reduced)
        else:
            out = self.model.decode(z, data.edge_label_index)
        out = out.view(-1).sigmoid()
        return out
