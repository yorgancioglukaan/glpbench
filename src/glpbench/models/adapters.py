"""Adapters exposing bundled models through the public model contract."""

from __future__ import annotations

from numbers import Real
from types import MappingProxyType
from typing import Mapping

import torch

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.exceptions import ConfigurationError, TrainingError
from glpbench.models.link_prediction import (
    AGAE_Model,
    DEALModel,
    FeatureCosineModel,
    FeatureMLPModel,
    GAE_Model,
    MLP_AGAE_Model,
    MLP_GAE_Model,
    VGAE_Model,
    available_agae_regularizers,
    get_model_registration,
)


_CONV_LAYER_TYPES = ("GCNConv", "GraphConv", "SAGEConv", "GATv2Conv")
_GRAPH_DEFAULTS = {
    "hidden_channels": 128,
    "out_channels": 64,
    "conv_layer_type": "GCNConv",
    "gatv2_kwargs": {},
}
_MLP_DEFAULTS = {
    "decoder_hidden_channels": None,
    "decoder_dropout": 0.0,
}
_AGAE_DEFAULTS = {
    "regularization_type": "none",
    "regularization_strength": 0.0,
}
_DEAL_DEFAULTS = {
    "hidden_channels": 128,
    "out_channels": 64,
    "theta_ss": 1.0,
    "theta_aa": 1.0,
    "theta_sa": 1.0,
    "lambda_ss": 1.0,
    "lambda_aa": 1.0,
    "lambda_sa": 1.0,
    "similarity_scale": 1.0,
    "similarity_margin": 0.0,
}


def _freeze(value):
    """Return a recursively immutable copy of a parameter value."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _mutable_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Copy a possibly immutable parameter mapping for a constructor."""
    result = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _mutable_mapping(item)
        else:
            result[key] = item
    return result


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(
            f"{name} must be a positive integer", component="model-adapter"
        )


def _number(value, name, *, minimum=None, positive=False):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ConfigurationError(
            f"{name} must be numeric", component="model-adapter"
        )
    if positive and value <= 0:
        raise ConfigurationError(
            f"{name} must be positive", component="model-adapter"
        )
    if minimum is not None and value < minimum:
        raise ConfigurationError(
            f"{name} must be at least {minimum}", component="model-adapter"
        )


class _BaseAdapter:
    """Share validation, loss delegation, and scoring across adapters."""

    name: str
    model_class: type[torch.nn.Module]
    defaults: Mapping[str, object] = MappingProxyType({})
    capabilities: ModelCapabilities

    def validate_parameters(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Validate and normalize model construction parameters."""
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(
                "model parameters must be a mapping",
                component="model-adapter",
                model=self.name,
            )
        unknown = set(parameters).difference(self.defaults)
        if unknown:
            fields = ", ".join(sorted(unknown))
            raise ConfigurationError(
                f"unsupported model parameter(s): {fields}",
                component="model-adapter",
                model=self.name,
            )
        normalized = {**self.defaults, **dict(parameters)}
        self._validate_values(normalized)
        return _freeze(normalized)

    def _validate_values(self, parameters: Mapping[str, object]) -> None:
        """Validate normalized values for one adapter."""

    def _constructor_parameters(
        self, train_data, parameters: Mapping[str, object]
    ) -> dict[str, object]:
        return _mutable_mapping(parameters)

    def build(self, train_data, parameters: Mapping[str, object]) -> torch.nn.Module:
        """Construct the wrapped model without creating an optimizer."""
        normalized = self.validate_parameters(parameters)
        constructor_parameters = self._constructor_parameters(
            train_data, normalized
        )
        return self.model_class(**constructor_parameters)

    def compute_training_loss(
        self,
        model: torch.nn.Module,
        train_data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> TrainingLoss:
        """Delegate loss computation and normalize its return type."""
        total, components = model.compute_training_loss(
            train_data, edge_label_index, edge_label, criterion
        )
        return TrainingLoss(total=total, components=components)

    @torch.no_grad()
    def score_edges(
        self, model: torch.nn.Module, data, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Score candidate edges using the existing model semantics."""
        model.eval()
        representations = model.encode(data.x, data.edge_index)
        scores = model.decode_edges(representations, edge_index, data).view(-1)
        if self.capabilities.output_kind != "probabilities":
            scores = scores.sigmoid()
        return scores

    def result_metadata(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Return immutable model-specific result fields."""
        self.validate_parameters(parameters)
        return MappingProxyType({})


class _GraphAdapter(_BaseAdapter):
    defaults = MappingProxyType(_GRAPH_DEFAULTS)

    def _validate_values(self, parameters):
        _positive_integer(parameters["hidden_channels"], "hidden_channels")
        _positive_integer(parameters["out_channels"], "out_channels")
        if parameters["conv_layer_type"] not in _CONV_LAYER_TYPES:
            raise ConfigurationError(
                f"conv_layer_type must be one of {_CONV_LAYER_TYPES}",
                component="model-adapter",
                model=self.name,
            )
        if not isinstance(parameters["gatv2_kwargs"], Mapping):
            raise ConfigurationError(
                "gatv2_kwargs must be a mapping",
                component="model-adapter",
                model=self.name,
            )

    def _constructor_parameters(self, train_data, parameters):
        values = _mutable_mapping(parameters)
        values["in_channels"] = train_data.num_features
        return values

    def result_metadata(self, parameters):
        normalized = self.validate_parameters(parameters)
        if normalized["conv_layer_type"] != "GATv2Conv":
            return MappingProxyType({})
        return _freeze({"gatv2_kwargs": normalized["gatv2_kwargs"]})


class _MLPMixin:
    def _validate_values(self, parameters):
        super()._validate_values(parameters)
        hidden = parameters["decoder_hidden_channels"]
        if hidden is not None:
            _positive_integer(hidden, "decoder_hidden_channels")
        _number(parameters["decoder_dropout"], "decoder_dropout", minimum=0)
        if parameters["decoder_dropout"] >= 1:
            raise ConfigurationError(
                "decoder_dropout must be less than 1",
                component="model-adapter",
                model=self.name,
            )

    def result_metadata(self, parameters):
        normalized = self.validate_parameters(parameters)
        metadata = dict(super().result_metadata(normalized))
        metadata.update({
            "decoder_hidden_channels": normalized["decoder_hidden_channels"],
            "decoder_dropout": normalized["decoder_dropout"],
        })
        return _freeze(metadata)


class _AGAEMixin:
    def _validate_values(self, parameters):
        super()._validate_values(parameters)
        if parameters["regularization_type"] not in available_agae_regularizers():
            raise ConfigurationError(
                "unknown AGAE regularization method",
                component="model-adapter",
                model=self.name,
            )
        _number(
            parameters["regularization_strength"],
            "regularization_strength",
            minimum=0,
        )

    def result_metadata(self, parameters):
        normalized = self.validate_parameters(parameters)
        metadata = dict(super().result_metadata(normalized))
        metadata.update({
            "regularization_type": normalized["regularization_type"],
            "regularization_strength": normalized["regularization_strength"],
        })
        return _freeze(metadata)


class GAEAdapter(_GraphAdapter):
    """Adapt the symmetric graph autoencoder."""

    name = "symmetric"
    model_class = GAE_Model
    capabilities = get_model_registration(name).capabilities


class GAEMLPAdapter(_MLPMixin, _GraphAdapter):
    """Adapt the symmetric graph autoencoder with an MLP decoder."""

    name = "mlp"
    model_class = MLP_GAE_Model
    defaults = MappingProxyType({**_GRAPH_DEFAULTS, **_MLP_DEFAULTS})
    capabilities = get_model_registration(name).capabilities


class AGAEAdapter(_AGAEMixin, _GraphAdapter):
    """Adapt the asymmetric graph autoencoder."""

    name = "asymmetric"
    model_class = AGAE_Model
    defaults = MappingProxyType({**_GRAPH_DEFAULTS, **_AGAE_DEFAULTS})
    capabilities = get_model_registration(name).capabilities


class AGAEMLPAdapter(_AGAEMixin, _MLPMixin, _GraphAdapter):
    """Adapt the asymmetric graph autoencoder with an MLP decoder."""

    name = "asymmetric-mlp"
    model_class = MLP_AGAE_Model
    defaults = MappingProxyType({
        **_GRAPH_DEFAULTS,
        **_MLP_DEFAULTS,
        **_AGAE_DEFAULTS,
    })
    capabilities = get_model_registration(name).capabilities


class VGAEAdapter(_GraphAdapter):
    """Adapt the variational graph autoencoder."""

    name = "variational"
    model_class = VGAE_Model
    defaults = MappingProxyType({**_GRAPH_DEFAULTS, "kl_weight": 1.0})
    capabilities = get_model_registration(name).capabilities

    def _validate_values(self, parameters):
        super()._validate_values(parameters)
        _number(parameters["kl_weight"], "kl_weight", minimum=0)

    def result_metadata(self, parameters):
        normalized = self.validate_parameters(parameters)
        metadata = dict(super().result_metadata(normalized))
        metadata["kl_weight"] = normalized["kl_weight"]
        return _freeze(metadata)


class DEALAdapter(_BaseAdapter):
    """Adapt the deep embedding alignment model."""

    name = "deal"
    model_class = DEALModel
    defaults = MappingProxyType(_DEAL_DEFAULTS)
    capabilities = get_model_registration(name).capabilities

    def _validate_values(self, parameters):
        _positive_integer(parameters["hidden_channels"], "hidden_channels")
        _positive_integer(parameters["out_channels"], "out_channels")
        theta = tuple(
            parameters[name] for name in ("theta_ss", "theta_aa", "theta_sa")
        )
        lambdas = tuple(
            parameters[name] for name in ("lambda_ss", "lambda_aa", "lambda_sa")
        )
        for name, value in zip(
            (
                "theta_ss",
                "theta_aa",
                "theta_sa",
                "lambda_ss",
                "lambda_aa",
                "lambda_sa",
            ),
            theta + lambdas,
        ):
            _number(value, name, minimum=0)
        if sum(theta) <= 0:
            raise ConfigurationError(
                "at least one theta weight must be positive",
                component="model-adapter",
                model=self.name,
            )
        if sum(lambdas) <= 0:
            raise ConfigurationError(
                "at least one lambda weight must be positive",
                component="model-adapter",
                model=self.name,
            )
        if parameters["lambda_aa"] + parameters["lambda_sa"] <= 0:
            raise ConfigurationError(
                "lambda_aa and lambda_sa cannot both be zero",
                component="model-adapter",
                model=self.name,
            )
        _number(parameters["similarity_scale"], "similarity_scale", positive=True)
        _number(parameters["similarity_margin"], "similarity_margin")

    def _constructor_parameters(self, train_data, parameters):
        values = _mutable_mapping(parameters)
        values.update({
            "in_channels": train_data.num_features,
            "train_data": train_data,
        })
        return values

    def result_metadata(self, parameters):
        normalized = self.validate_parameters(parameters)
        return _freeze({
            f"deal_{name}": normalized[name]
            for name in (
                "theta_ss", "theta_aa", "theta_sa", "lambda_ss", "lambda_aa",
                "lambda_sa", "similarity_scale", "similarity_margin",
            )
        })


class FeatureCosineAdapter(_BaseAdapter):
    """Adapt parameter-free feature cosine similarity."""

    name = "feature-cosine"
    model_class = FeatureCosineModel
    capabilities = get_model_registration(name).capabilities

    def compute_training_loss(
        self, model, train_data, edge_label_index, edge_label, criterion
    ):
        """Reject optimization for the parameter-free baseline."""
        raise TrainingError(
            "feature-cosine is not trainable",
            component="model-adapter",
            model=self.name,
        )


class FeatureMLPAdapter(_MLPMixin, _BaseAdapter):
    """Adapt the trainable feature-only MLP."""

    name = "feature-mlp"
    model_class = FeatureMLPModel
    defaults = MappingProxyType(_MLP_DEFAULTS)
    capabilities = get_model_registration(name).capabilities

    def _constructor_parameters(self, train_data, parameters):
        values = _mutable_mapping(parameters)
        values["in_channels"] = train_data.num_features
        return values


__all__ = [
    "AGAEAdapter",
    "AGAEMLPAdapter",
    "DEALAdapter",
    "FeatureCosineAdapter",
    "FeatureMLPAdapter",
    "GAEAdapter",
    "GAEMLPAdapter",
    "VGAEAdapter",
]
