"""Reusable base class for model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

import torch

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.exceptions import ConfigurationError


def _freeze(value):
    """Return a recursively immutable copy of a parameter value."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class BaseModelAdapter(ABC):
    """Provide common validation, loss, and scoring for model adapters.

    Subclasses declare ``name``, ``capabilities``, and ``defaults``, and
    implement ``build``. Override the other hooks when model behavior differs.
    Models using the supplied loss and scoring methods implement
    ``compute_training_loss``, ``encode``, and ``decode_edges``.
    """

    name: str | None = None
    capabilities: ModelCapabilities | None = None
    defaults: Mapping[str, object] | None = None

    def __init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("adapter must declare a nonempty name")
        if not isinstance(self.capabilities, ModelCapabilities):
            raise TypeError("adapter must declare ModelCapabilities")
        if not isinstance(self.defaults, Mapping):
            raise TypeError("adapter must declare a defaults mapping")
        self._parameter_defaults = _freeze(self.defaults)

    @property
    def parameter_defaults(self) -> Mapping[str, object]:
        """Return immutable defaults defining accepted parameters."""
        return self._parameter_defaults

    def validate_parameters(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Merge defaults and validate adapter-specific values."""
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(
                "model parameters must be a mapping",
                component="model-adapter",
                model=self.name,
            )
        unknown = set(parameters).difference(self.parameter_defaults)
        if unknown:
            fields = ", ".join(sorted(unknown))
            raise ConfigurationError(
                f"unsupported model parameter(s): {fields}",
                component="model-adapter",
                model=self.name,
            )
        normalized = {**self.parameter_defaults, **dict(parameters)}
        self._validate_values(normalized)
        return _freeze(normalized)

    def _validate_values(self, parameters: Mapping[str, object]) -> None:
        """Validate values after defaults are applied; override as needed."""

    @abstractmethod
    def build(
        self, train_data, parameters: Mapping[str, object]
    ) -> torch.nn.Module:
        """Construct a fresh model without training it."""

    def compute_training_loss(
        self,
        model: torch.nn.Module,
        train_data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> TrainingLoss:
        """Delegate one training loss to a compatible model."""
        total, components = model.compute_training_loss(
            train_data, edge_label_index, edge_label, criterion
        )
        return TrainingLoss(total=total, components=components)

    @torch.no_grad()
    def score_edges(
        self, model: torch.nn.Module, data, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode and decode candidates using existing score semantics."""
        model.eval()
        representations = model.encode(data.x, data.edge_index)
        scores = model.decode_edges(representations, edge_index, data).view(-1)
        if self.capabilities.output_kind != "probabilities":
            scores = scores.sigmoid()
        return scores

    def result_metadata(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Return empty metadata after parameter validation."""
        self.validate_parameters(parameters)
        return MappingProxyType({})


__all__ = ["BaseModelAdapter"]
