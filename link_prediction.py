from dataclasses import dataclass
import random

from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, SAGEConv
import torch
import torch.nn.functional as F
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


class LinkPredictionModel(torch.nn.Module):
    """Base interface for link-prediction model architectures."""

    def decode_edges(self, z, edge_label_index, data):
        """Score labeled edges using any model-specific data."""
        raise NotImplementedError

    def reconstruction_loss(self, logits, labels, criterion):
        """Calculate the edge reconstruction loss."""
        return criterion(logits, labels)

    def regularization_loss(self):
        """Return an optional architecture-specific regularization term."""
        return next(self.parameters()).new_zeros(())

    def compute_training_loss(
        self, data, edge_label_index, edge_label, criterion
    ):
        """Return the total training loss and named loss components."""
        z = self.encode(data.x, data.edge_index)
        logits = self.decode_edges(z, edge_label_index, data).view(-1)
        reconstruction = self.reconstruction_loss(logits, edge_label, criterion)
        regularization = self.regularization_loss()
        return reconstruction + regularization, {
            "reconstruction": reconstruction,
            "regularization": regularization,
        }


class AGAE_Model(LinkPredictionModel):
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

    def decode_edges(self, z, edge_label_index, data):
        """Score edges using reduced features from a graph split."""
        return self.decode(z, edge_label_index, data.x_reduced)

    def decode_all(self, z, x_reduced):
        """Return node pairs with reconstruction scores above 0.5."""
        prob_adj = z @ x_reduced.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()

class GAE_Model(LinkPredictionModel):
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

    def decode_edges(self, z, edge_label_index, data):
        """Score edges from latent representations."""
        return self.decode(z, edge_label_index)

    def decode_all(self, z):
        """Return node pairs with reconstruction scores above 0.5."""
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0.5).nonzero(as_tuple=False).t()


class MLP_GAE_Model(GAE_Model):
    """Graph autoencoder with a learned symmetric MLP decoder."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type=GCNConv,
        decoder_hidden_channels=None,
        decoder_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            conv_layer_type=conv_layer_type,
            **kwargs,
        )
        if decoder_hidden_channels is None:
            decoder_hidden_channels = out_channels
        if not isinstance(decoder_hidden_channels, int) or decoder_hidden_channels <= 0:
            raise ValueError("decoder_hidden_channels must be a positive integer")
        if not isinstance(decoder_dropout, (int, float)) or not 0 <= decoder_dropout < 1:
            raise ValueError("decoder_dropout must be between 0 and 1")
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels, decoder_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(decoder_dropout),
            torch.nn.Linear(decoder_hidden_channels, 1),
        )

    def decode(self, z, edge_label_index):
        """Score edges from symmetric pairwise latent features."""
        source = z[edge_label_index[0]]
        target = z[edge_label_index[1]]
        edge_features = torch.cat(
            [source * target, torch.abs(source - target)], dim=-1
        )
        return self.edge_decoder(edge_features).squeeze(-1)

    def decode_all(self, z):
        """Return node pairs with decoder logits above 0.5."""
        nodes = torch.arange(z.size(0), device=z.device)
        edge_index = torch.cartesian_prod(nodes, nodes).t()
        return edge_index[:, self.decode(z, edge_index) > 0.5]


class VGAE_Model(LinkPredictionModel):
    """Variational graph autoencoder with an inner-product decoder."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type=GCNConv,
        kl_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(kl_weight, (int, float)) or kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        self.kl_weight = kl_weight

        layer_types = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
        }
        if conv_layer_type in layer_types:
            layer_class = layer_types[conv_layer_type]
            self.conv1 = layer_class(in_channels, hidden_channels)
            self.conv_mu = layer_class(hidden_channels, out_channels)
            self.conv_logstd = layer_class(hidden_channels, out_channels)
        elif conv_layer_type == "GATv2Conv":
            self.conv1, self.conv_mu = _gatv2_layers(
                in_channels, hidden_channels, out_channels, kwargs
            )
            _, self.conv_logstd = _gatv2_layers(
                in_channels, hidden_channels, out_channels, kwargs
            )
        else:
            raise ValueError(
                "conv_layer_type must be one of 'GCNConv', 'GraphConv', "
                "'SAGEConv', 'GATv2Conv'"
            )
        self.mu = None
        self.logstd = None

    def encode(self, x, edge_index):
        """Sample latent nodes in training and return means in evaluation."""
        hidden = self.conv1(x, edge_index).relu()
        self.mu = self.conv_mu(hidden, edge_index)
        self.logstd = self.conv_logstd(hidden, edge_index).clamp(max=10)
        if self.training:
            return self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)
        return self.mu

    def decode(self, z, edge_label_index):
        """Score edges using pairwise latent dot products."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_edges(self, z, edge_label_index, data):
        """Score edges from latent representations."""
        return self.decode(z, edge_label_index)

    def decode_all(self, z):
        """Return node pairs with reconstruction scores above 0.5."""
        return ((z @ z.t()) > 0.5).nonzero(as_tuple=False).t()

    def regularization_loss(self):
        """Return weighted KL divergence from a unit Gaussian prior."""
        if self.mu is None or self.logstd is None:
            raise RuntimeError("encode must be called before regularization_loss")
        kl_loss = -0.5 * torch.mean(torch.sum(
            1 + 2 * self.logstd - self.mu.pow(2) - torch.exp(2 * self.logstd),
            dim=1,
        ))
        return self.kl_weight * kl_loss


class FeatureCosineModel(torch.nn.Module):
    """Score edges using cosine similarity between node features."""

    def encode(self, x, edge_index):
        """Return node features without message passing."""
        return x

    def decode_edges(self, features, edge_label_index, data):
        """Return cosine similarities mapped to the unit interval."""
        source = features[edge_label_index[0]]
        target = features[edge_label_index[1]]
        similarity = F.cosine_similarity(source, target, dim=-1, eps=1e-8)
        return (similarity + 1.0) / 2.0


@dataclass(frozen=True)
class ModelSpec:
    """Describe construction and training requirements for an architecture."""

    constructor: type[torch.nn.Module]
    trainable: bool = True
    uses_conv_layer: bool = True
    outputs_probabilities: bool = False
    uses_graph_context: bool = False


_MODEL_REGISTRY = {
    "symmetric": ModelSpec(GAE_Model),
    "asymmetric": ModelSpec(AGAE_Model),
    "mlp": ModelSpec(MLP_GAE_Model),
    "variational": ModelSpec(VGAE_Model),
    "feature-cosine": ModelSpec(
        FeatureCosineModel,
        trainable=False,
        uses_conv_layer=False,
        outputs_probabilities=True,
    ),
}


def available_model_types():
    """Return the registered link-prediction architecture names."""
    return tuple(_MODEL_REGISTRY)


def sample_training_negative_edges(data, seed=None):
    """Sample training negatives, optionally without changing RNG state."""
    def sample_edges():
        return negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1),
        )

    if seed is None:
        return sample_edges()

    devices = []
    if data.edge_index.device.type == "cuda":
        devices = [data.edge_index.device.index]
    python_random_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            torch.manual_seed(seed)
            return sample_edges()
    finally:
        random.setstate(python_random_state)


def _validate_negative_edge_index(edge_index, num_nodes, device):
    """Validate externally supplied negative edges."""
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("negative_edge_index must be a torch.Tensor")
    if edge_index.dtype != torch.long:
        raise TypeError("negative_edge_index must have dtype torch.long")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("negative_edge_index must have shape [2, num_edges]")
    if edge_index.device != device:
        raise ValueError("negative_edge_index must be on the training data device")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("negative_edge_index contains an invalid node ID")


class LinkPrediction:
    """Train and evaluate graph-autoencoder link predictors.

    Parameters
    ----------
    train_data : torch_geometric.data.Data
        Graph used for training.

    Attributes
    ----------
    model_type : str
        Registered link-prediction architecture.
    optimizer : torch.optim.Optimizer or None
        Model optimizer, or ``None`` for non-trainable predictors.
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

        if self.model_type not in _MODEL_REGISTRY:
             raise ValueError("Unexpected model type: " + self.model_type)
        self.model_spec = _MODEL_REGISTRY[self.model_type]
        self.trainable = self.model_spec.trainable
        if self.model_spec.uses_conv_layer or self.model_spec.uses_graph_context:
            constructor_kwargs = {
                "in_channels": train_data.num_features,
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                **model_kwargs,
            }
            if self.model_spec.uses_conv_layer:
                constructor_kwargs["conv_layer_type"] = conv_layer_type
            if self.model_spec.uses_graph_context:
                constructor_kwargs["train_data"] = train_data
            self.model = self.model_spec.constructor(
                **constructor_kwargs,
            )
        else:
            self.model = self.model_spec.constructor()

        self.optimizer = (
            torch.optim.Adam(params=self.model.parameters(), lr=0.01)
            if self.trainable else None
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, train_data=None, negative_edge_index=None):
        """Train the model for one epoch.

        Parameters
        ----------
        train_data : torch_geometric.data.Data, optional
            Training graph. Defaults to the graph supplied at construction.
        negative_edge_index : torch.Tensor, optional
            Precomputed negative edges. By default, negatives are sampled.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """

        if not self.trainable:
            raise RuntimeError(f"{self.model_type} is not trainable")

        if train_data == None:
             train_data = self.train_data


        self.model.train()
        self.optimizer.zero_grad()

        if negative_edge_index is None:
            negative_edge_index = sample_training_negative_edges(train_data)
        _validate_negative_edge_index(
            negative_edge_index,
            train_data.num_nodes,
            train_data.edge_label_index.device,
        )

        edge_label_index = torch.cat(
            [train_data.edge_label_index, negative_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(negative_edge_index.size(1))
        ], dim=0)

        loss, components = self.model.compute_training_loss(
            train_data, edge_label_index, edge_label, self.criterion
        )
        if not isinstance(components, dict):
            raise TypeError("training loss components must be a dictionary")
        self.last_loss_components = components
        zero = loss.new_zeros(())
        self.last_reconstruction_loss = components.get("reconstruction", loss)
        self.last_regularization_loss = components.get("regularization", zero)
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
        out = self.model.decode_edges(z, data.edge_label_index, data)
        out = out.view(-1)
        if not self.model_spec.outputs_probabilities:
            out = out.sigmoid()
        return out
