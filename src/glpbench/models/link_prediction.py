from collections.abc import Mapping
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


def _regularize_none(model, data, z):
    """Return a zero penalty."""
    return z.new_zeros(())


def _seen_representations(data, z):
    """Select representations belonging to training-visible nodes."""
    visible_node_mask = getattr(data, "visible_node_mask", None)
    return z if visible_node_mask is None else z[visible_node_mask]


def _regularize_latent_l2(model, data, z):
    """Penalize squared latent-vector norms on visible nodes."""
    return _seen_representations(data, z).square().sum(dim=-1).mean()


def _aligned_seen_representations(data, z):
    """Return dimensionally compatible latent and reduced features."""
    if not hasattr(data, "x_reduced"):
        raise ValueError("alignment regularization requires data.x_reduced")
    if z.size(-1) != data.x_reduced.size(-1):
        raise ValueError("z and x_reduced must have the same feature dimension")
    visible_node_mask = getattr(data, "visible_node_mask", None)
    if visible_node_mask is None:
        return z, data.x_reduced
    return z[visible_node_mask], data.x_reduced[visible_node_mask]


def _regularize_mse_alignment(model, data, z):
    """Penalize squared distance from reduced features on visible nodes."""
    latent, reduced = _aligned_seen_representations(data, z)
    return F.mse_loss(latent, reduced)


def _regularize_cosine_alignment(model, data, z):
    """Penalize cosine distance from reduced features on visible nodes."""
    latent, reduced = _aligned_seen_representations(data, z)
    return 1.0 - F.cosine_similarity(
        latent, reduced, dim=-1, eps=1e-8
    ).mean()


def _regularize_graph_smoothness(model, data, z):
    """Penalize latent differences across training graph edges."""
    edge_index = data.edge_index
    visible_node_mask = getattr(data, "visible_node_mask", None)
    if visible_node_mask is not None:
        seen_edges = visible_node_mask[edge_index].all(dim=0)
        edge_index = edge_index[:, seen_edges]
    if edge_index.size(1) == 0:
        return z.new_zeros(())
    source, target = edge_index
    return (z[source] - z[target]).square().sum(dim=-1).mean()


_AGAE_REGULARIZERS = {
    "none": _regularize_none,
    "latent-l2": _regularize_latent_l2,
    "mse-alignment": _regularize_mse_alignment,
    "cosine-alignment": _regularize_cosine_alignment,
    "graph-smoothness": _regularize_graph_smoothness,
}


def available_agae_regularizers():
    """Return registered AGAE regularization methods."""
    return tuple(_AGAE_REGULARIZERS)


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
    regularization_type : str, optional
        Registered AGAE regularizer.
    regularization_strength : float, optional
        Non-negative multiplier applied to the regularization term.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type="GCNConv",
        regularization_type="none",
        regularization_strength=0.0,
        **kwargs,
    ):
        super().__init__()

        if regularization_type not in _AGAE_REGULARIZERS:
            raise ValueError(f"Unknown AGAE regularization: {regularization_type}")
        if not isinstance(regularization_strength, (int, float)):
            raise TypeError("regularization_strength must be numeric")
        if regularization_strength < 0:
            raise ValueError("regularization_strength must be non-negative")
        self.regularization_type = regularization_type
        self.regularization_strength = float(regularization_strength)

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

    def compute_training_loss(
        self, data, edge_label_index, edge_label, criterion
    ):
        """Return reconstruction plus the configured regularization loss."""
        z = self.encode(data.x, data.edge_index)
        logits = self.decode_edges(z, edge_label_index, data).view(-1)
        reconstruction = criterion(logits, edge_label)
        raw_regularization = _AGAE_REGULARIZERS[self.regularization_type](
            self, data, z
        )
        regularization = self.regularization_strength * raw_regularization
        return reconstruction + regularization, {
            "reconstruction": reconstruction,
            "regularization": regularization,
            "raw_regularization": raw_regularization,
        }

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

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type="GCNConv",
        **kwargs,
    ):
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


def _make_mlp_decoder(out_channels, hidden_channels, dropout):
    """Build and validate the shared learned edge decoder."""
    if hidden_channels is None:
        hidden_channels = out_channels
    if not isinstance(hidden_channels, int) or hidden_channels <= 0:
        raise ValueError("decoder_hidden_channels must be a positive integer")
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("decoder_dropout must be between 0 and 1")
    return torch.nn.Sequential(
        torch.nn.Linear(2 * out_channels, hidden_channels),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_channels, 1),
    )


class MLP_GAE_Model(GAE_Model):
    """Graph autoencoder with a learned symmetric MLP decoder.

    ``decoder_hidden_channels`` sets the edge-decoder width and defaults to
    ``out_channels``. ``decoder_dropout`` is applied between decoder layers.
    Encoder arguments match :class:`GAE_Model`.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type="GCNConv",
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
        self.edge_decoder = _make_mlp_decoder(
            out_channels, decoder_hidden_channels, decoder_dropout
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


class MLP_AGAE_Model(AGAE_Model):
    """Asymmetric graph autoencoder with a learned MLP decoder.

    The decoder combines source latent vectors and target reduced features
    using elementwise products and absolute differences. Decoder arguments
    match :class:`MLP_GAE_Model`; encoder and regularization arguments match
    :class:`AGAE_Model`.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type="GCNConv",
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
        self.edge_decoder = _make_mlp_decoder(
            out_channels, decoder_hidden_channels, decoder_dropout
        )

    def decode(self, z, edge_label_index, x_reduced):
        """Score edges from latent sources and feature-based targets."""
        source = z[edge_label_index[0]]
        target = x_reduced[edge_label_index[1]]
        edge_features = torch.cat(
            [source * target, torch.abs(source - target)], dim=-1
        )
        return self.edge_decoder(edge_features).squeeze(-1)

    def decode_all(self, z, x_reduced):
        """Return node pairs with decoder logits above 0.5."""
        nodes = torch.arange(z.size(0), device=z.device)
        edge_index = torch.cartesian_prod(nodes, nodes).t()
        return edge_index[:, self.decode(z, edge_index, x_reduced) > 0.5]


class VGAE_Model(LinkPredictionModel):
    """Variational graph autoencoder with an inner-product decoder.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Input, hidden, and latent feature dimensions.
    conv_layer_type : str, optional
        Message-passing layer type.
    kl_weight : float, optional
        Non-negative multiplier for KL divergence.
    **kwargs
        GATv2 constructor options when ``conv_layer_type="GATv2Conv"``.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        conv_layer_type="GCNConv",
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


class DEALModel(LinkPredictionModel):
    """Deep embedding alignment model for attributed link prediction.

    The structure branch embeds visible node identities, while the attribute
    branch maps node features into the same latent space. Query nodes have no
    trainable structural embedding and are scored through their attributes.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Input, attribute-hidden, and shared embedding dimensions.
    train_data : torch_geometric.data.Data
        Training graph with Boolean validation-query, test-query, and
        visible-node masks.
    theta_ss, theta_aa, theta_sa : float, optional
        Non-negative training-loss weights for structural, attribute, and
        cross-branch scores.
    lambda_ss, lambda_aa, lambda_sa : float, optional
        Non-negative inference weights for the same branches.
    similarity_scale : float, optional
        Positive scale applied to cosine-similarity logits.
    similarity_margin : float, optional
        Margin subtracted from similarity logits.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        train_data,
        theta_ss=1.0,
        theta_aa=1.0,
        theta_sa=1.0,
        lambda_ss=1.0,
        lambda_aa=1.0,
        lambda_sa=1.0,
        similarity_scale=1.0,
        similarity_margin=0.0,
        **kwargs,
    ):
        super().__init__()
        self._validate_weights(
            "theta", (theta_ss, theta_aa, theta_sa), require_all=False
        )
        self._validate_weights(
            "lambda", (lambda_ss, lambda_aa, lambda_sa), require_all=False
        )
        if lambda_aa + lambda_sa <= 0:
            raise ValueError("lambda_aa and lambda_sa cannot both be zero")
        if not isinstance(similarity_scale, (int, float)) or similarity_scale <= 0:
            raise ValueError("similarity_scale must be positive")
        if not isinstance(similarity_margin, (int, float)):
            raise TypeError("similarity_margin must be numeric")

        validation_query_mask = getattr(
            train_data, "validation_query_mask", None
        )
        test_query_mask = getattr(train_data, "test_query_mask", None)
        visible_node_mask = getattr(train_data, "visible_node_mask", None)
        for name, mask in (
            ("validation_query_mask", validation_query_mask),
            ("test_query_mask", test_query_mask),
            ("visible_node_mask", visible_node_mask),
        ):
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ValueError(f"train_data.{name} must be a boolean tensor")
            if mask.ndim != 1 or mask.numel() != train_data.num_nodes:
                raise ValueError(
                    f"train_data.{name} must have one entry per node"
                )
        query_node_mask = validation_query_mask | test_query_mask
        if not torch.equal(query_node_mask, ~visible_node_mask):
            raise ValueError("query and visible node masks must be complements")

        visible_node_ids = visible_node_mask.nonzero(as_tuple=False).view(-1)
        if visible_node_ids.numel() == 0:
            raise ValueError("DEAL requires at least one visible node")
        global_to_visible = torch.full(
            (train_data.num_nodes,), -1, dtype=torch.long,
            device=visible_node_ids.device,
        )
        global_to_visible[visible_node_ids] = torch.arange(
            visible_node_ids.numel(), device=visible_node_ids.device
        )
        self.register_buffer("query_node_mask", query_node_mask.clone())
        self.register_buffer("visible_node_mask", visible_node_mask.clone())
        self.register_buffer("visible_node_ids", visible_node_ids)
        self.register_buffer("global_to_visible", global_to_visible)

        self.structure_embeddings = torch.nn.Embedding(
            visible_node_ids.numel(), out_channels
        )
        self.attribute_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.ELU(),
        )
        self.theta = (float(theta_ss), float(theta_aa), float(theta_sa))
        self.lambdas = (float(lambda_ss), float(lambda_aa), float(lambda_sa))
        self.similarity_scale = float(similarity_scale)
        self.similarity_margin = float(similarity_margin)

    @staticmethod
    def _validate_weights(name, weights, require_all=False):
        """Validate non-negative loss or score weights."""
        if any(not isinstance(value, (int, float)) for value in weights):
            raise TypeError(f"{name} weights must be numeric")
        if any(value < 0 for value in weights):
            raise ValueError(f"{name} weights must be non-negative")
        if require_all and any(value == 0 for value in weights):
            raise ValueError(f"{name} weights must be positive")
        if sum(weights) <= 0:
            raise ValueError(f"at least one {name} weight must be positive")

    def encode(self, x, edge_index):
        """Return structural and attribute representations for all nodes."""
        structure = x.new_zeros((x.size(0), self.structure_embeddings.embedding_dim))
        structure[self.visible_node_ids] = self.structure_embeddings.weight
        return structure, self.attribute_encoder(x)

    def _similarity(self, source, target):
        """Return scaled cosine-similarity logits."""
        similarity = F.cosine_similarity(source, target, dim=-1, eps=1e-8)
        return self.similarity_scale * similarity - self.similarity_margin

    def _component_logits(self, representations, edge_label_index):
        """Return structure, attribute, and alignment logits."""
        structure, attributes = representations
        source, target = edge_label_index
        return (
            self._similarity(structure[source], structure[target]),
            self._similarity(attributes[source], attributes[target]),
            self._similarity(structure[source], attributes[target]),
        )

    @staticmethod
    def _weighted_mean(values, weights):
        """Combine tensors using normalized non-negative weights."""
        total_weight = sum(weights)
        return sum(weight * value for weight, value in zip(weights, values)) / total_weight

    def compute_training_loss(
        self, data, edge_label_index, edge_label, criterion
    ):
        """Calculate DEAL's structure, attribute, and alignment losses."""
        if not bool(self.visible_node_mask[edge_label_index].all()):
            raise ValueError("DEAL training edges must contain only visible nodes")
        representations = self.encode(data.x, data.edge_index)
        logits = self._component_logits(representations, edge_label_index)
        losses = tuple(criterion(component, edge_label) for component in logits)
        total = self._weighted_mean(losses, self.theta)
        return total, {
            "structure": losses[0],
            "attribute": losses[1],
            "alignment": losses[2],
        }

    def decode_edges(self, representations, edge_label_index, data):
        """Score transductive or inductive edges with valid DEAL branches."""
        component_logits = self._component_logits(representations, edge_label_index)
        split_setting = getattr(data, "split_setting", None)
        if split_setting == "transductive":
            if not bool(self.visible_node_mask[edge_label_index].all()):
                raise ValueError("transductive DEAL edges must contain only visible nodes")
            return self._weighted_mean(component_logits, self.lambdas)
        if split_setting == "semi-inductive":
            source, target = edge_label_index
            if not bool(self.visible_node_mask[source].all()) or not bool(
                self.query_node_mask[target].all()
            ):
                raise ValueError(
                    "inductive DEAL edges must be oriented from visible to query nodes"
                )
            return self._weighted_mean(
                component_logits[1:], self.lambdas[1:]
            )
        raise ValueError(
            "DEAL requires a transductive or semi-inductive split setting"
        )


class FeatureCosineModel(torch.nn.Module):
    """Score edges with parameter-free cosine similarity of node features.

    Similarities in ``[-1, 1]`` are mapped to probabilities in ``[0, 1]``.
    The graph topology is ignored.
    """

    def encode(self, x, edge_index):
        """Return node features without message passing."""
        return x

    def decode_edges(self, features, edge_label_index, data):
        """Return cosine similarities mapped to the unit interval."""
        source = features[edge_label_index[0]]
        target = features[edge_label_index[1]]
        similarity = F.cosine_similarity(source, target, dim=-1, eps=1e-8)
        return (similarity + 1.0) / 2.0


class FeatureMLPModel(LinkPredictionModel):
    """Score node pairs with a trainable symmetric feature MLP.

    Parameters
    ----------
    in_channels : int
        Input node-feature dimension.
    decoder_hidden_channels : int, optional
        Edge-decoder width. Defaults to ``in_channels``.
    decoder_dropout : float, optional
        Dropout probability between decoder layers.

    Notes
    -----
    The graph topology is ignored. Pair features concatenate the elementwise
    product and absolute difference of endpoint features.
    """

    def __init__(
        self,
        in_channels,
        decoder_hidden_channels=None,
        decoder_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.edge_decoder = _make_mlp_decoder(
            in_channels, decoder_hidden_channels, decoder_dropout
        )

    def encode(self, x, edge_index):
        """Return node features without message passing."""
        return x

    def decode_edges(self, features, edge_label_index, data):
        """Score pairs using symmetric elementwise feature interactions."""
        source = features[edge_label_index[0]]
        target = features[edge_label_index[1]]
        edge_features = torch.cat(
            [source * target, torch.abs(source - target)], dim=-1
        )
        return self.edge_decoder(edge_features).squeeze(-1)


def available_model_types():
    """Return the registered link-prediction architecture names."""
    from glpbench.models.registry import available_model_types as available

    return available()


def count_trainable_parameters(model):
    """Return the number of trainable parameters in a PyTorch model."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def sample_training_negative_edges(data, seed=None):
    """Sample one negative edge per positive training edge.

    When ``visible_node_mask`` is present, both endpoints are restricted to
    visible nodes. Supplying ``seed`` makes sampling reproducible without changing the
    caller's Python or PyTorch random-number-generator state.

    Returns
    -------
    torch.Tensor
        Negative edge index with shape ``[2, num_training_positives]``.
    """
    def sample_edges():
        visible_node_mask = getattr(data, "visible_node_mask", None)
        if visible_node_mask is None:
            return negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.edge_label_index.size(1),
            )

        visible_node_ids = visible_node_mask.nonzero(as_tuple=False).view(-1)
        global_to_visible = torch.full(
            (data.num_nodes,), -1, dtype=torch.long, device=data.edge_index.device
        )
        global_to_visible[visible_node_ids] = torch.arange(
            visible_node_ids.numel(), device=data.edge_index.device
        )
        local_edge_index = global_to_visible[data.edge_index]
        if bool((local_edge_index < 0).any()):
            raise ValueError("training edge_index must contain only visible nodes")
        local_negatives = negative_sampling(
            edge_index=local_edge_index,
            num_nodes=visible_node_ids.numel(),
            num_neg_samples=data.edge_label_index.size(1),
        )
        return visible_node_ids[local_negatives]

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


_LEGACY_MODEL_PARAMETERS = {
    "decoder_hidden_channels",
    "decoder_dropout",
    "gatv2_kwargs",
    "kl_weight",
    "lambda_aa",
    "lambda_sa",
    "lambda_ss",
    "regularization_strength",
    "regularization_type",
    "similarity_margin",
    "similarity_scale",
    "theta_aa",
    "theta_sa",
    "theta_ss",
    *_GATV2_ARGUMENTS,
}


class LinkPrediction:
    """Train and evaluate graph-autoencoder link predictors.

    Parameters
    ----------
    train_data : torch_geometric.data.Data
        Graph used for training. It must contain ``x``, ``edge_index``,
        ``edge_label_index``, and ``edge_label``. Reduced-feature models also
        require ``x_reduced``.
    model_type : str, optional
        Registered architecture name. Defaults to ``"gae"``.
    hidden_channels, out_channels : int, optional
        Encoder hidden and latent dimensions.
    conv_layer_type : str, optional
        Message-passing layer name for graph encoders.
    learning_rate, weight_decay : float, optional
        Adam optimizer settings for trainable models.
    model_parameters : mapping, optional
        Explicit adapter parameters. This cannot be combined with legacy
        model-construction keyword arguments.
    **kwargs
        Architecture-specific decoder, regularization, VGAE, DEAL, or GATv2
        arguments.

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
            Model, optimizer, encoder, decoder, and architecture-specific
            options described by :class:`LinkPrediction`.
        """

        self.train_data = train_data
        requested_model_type = kwargs.get("model_type", "gae")
        explicit_model_parameters = kwargs.get("model_parameters")
        hidden_channels = kwargs.get("hidden_channels", 128)
        out_channels = kwargs.get("out_channels", 64)
        conv_layer_type = kwargs.get("conv_layer_type", "GCNConv")
        learning_rate = kwargs.get("learning_rate", 0.01)
        weight_decay = kwargs.get("weight_decay", 0.0)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        model_kwargs = {
            name: value for name, value in kwargs.items()
            if name not in {
                "model_type", "hidden_channels", "out_channels", "conv_layer_type",
                "learning_rate", "weight_decay", "model_parameters",
            }
        }

        from glpbench.models.registry import get_model_adapter

        self.model_adapter = get_model_adapter(requested_model_type)
        self.model_type = self.model_adapter.name
        capabilities = self.model_adapter.capabilities
        self.trainable = capabilities.trainable
        adapter_defaults = self.model_adapter.parameter_defaults
        if explicit_model_parameters is not None:
            if not isinstance(explicit_model_parameters, Mapping):
                raise TypeError("model_parameters must be a mapping")
            conflicting = set(model_kwargs)
            if isinstance(adapter_defaults, Mapping):
                conflicting.update(
                    name for name in (
                        "hidden_channels", "out_channels", "conv_layer_type"
                    )
                    if name in kwargs and name in adapter_defaults
                )
            if conflicting:
                fields = ", ".join(sorted(conflicting))
                raise ValueError(
                    "model_parameters cannot be combined with legacy model "
                    f"arguments: {fields}"
                )
            model_parameters = dict(explicit_model_parameters)
        elif isinstance(adapter_defaults, Mapping):
            if conv_layer_type == "GATv2Conv" and any(
                name in model_kwargs for name in _GATV2_ARGUMENTS
            ):
                gatv2_kwargs = model_kwargs.get("gatv2_kwargs", {})
                if isinstance(gatv2_kwargs, Mapping):
                    gatv2_kwargs = dict(gatv2_kwargs)
                    gatv2_kwargs.update({
                        name: model_kwargs[name]
                        for name in _GATV2_ARGUMENTS
                        if name in model_kwargs
                    })
                    model_kwargs["gatv2_kwargs"] = gatv2_kwargs
            candidate_parameters = {
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                "conv_layer_type": conv_layer_type,
                **model_kwargs,
            }
            model_parameters = {
                name: value for name, value in candidate_parameters.items()
                if name in adapter_defaults
            }
            unknown_parameters = {
                name: value for name, value in model_kwargs.items()
                if name not in _LEGACY_MODEL_PARAMETERS
            }
            model_parameters.update(unknown_parameters)
        else:
            model_parameters = dict(model_kwargs)
            for name, value in (
                ("hidden_channels", hidden_channels),
                ("out_channels", out_channels),
                ("conv_layer_type", conv_layer_type),
            ):
                if name in kwargs:
                    model_parameters[name] = value
        self.model_parameters = self.model_adapter.validate_parameters(
            model_parameters
        )
        self.model = self.model_adapter.build(train_data, self.model_parameters)

        self.optimizer = (
            torch.optim.Adam(
                params=self.model.parameters(),
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
            )
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

        if train_data is None:
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

        training_loss = self.model_adapter.compute_training_loss(
            self.model,
            train_data,
            edge_label_index,
            edge_label,
            self.criterion,
        )
        loss = training_loss.total
        components = dict(training_loss.components)
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
        return self.model_adapter.score_edges(
            self.model, data, data.edge_label_index
        )
