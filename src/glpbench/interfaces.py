"""Structural interfaces for extensible GLPBench components."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch

from glpbench.contracts import (
    AuditReport,
    EvaluationResult,
    GraphData,
    LoadedDataset,
    ModelCapabilities,
    ProtocolRequirements,
    ResolvedExperimentSpec,
    SplitBundle,
    TrainingLoss,
    TrainingResult,
)


@runtime_checkable
class DatasetProvider(Protocol):
    """Load a named dataset into the standard evaluation envelope."""

    name: str

    def load(self, root: Path, device: str | torch.device) -> LoadedDataset:
        """Load and standardize the dataset."""
        ...


@runtime_checkable
class SplitProtocol(Protocol):
    """Construct and audit one reproducible graph-splitting protocol."""

    name: str
    requirements: ProtocolRequirements

    def split(
        self,
        data: GraphData,
        seed: int,
        *,
        dataset_fingerprint: str | None = None,
    ) -> SplitBundle:
        """Create training, validation, and test partitions."""
        ...

    def audit(self, data: GraphData, splits: SplitBundle) -> AuditReport:
        """Check protocol invariants for a realized split."""
        ...


@runtime_checkable
class FeaturePreprocessor(Protocol):
    """Fit and apply a node-feature transformation."""

    name: str
    requires_fitting: bool
    supports_sparse_inputs: bool
    uses_randomness: bool

    def fit(
        self,
        x: torch.Tensor,
        fit_mask: torch.Tensor,
        seed: int | None = None,
    ) -> "FeaturePreprocessor":
        """Fit using only nodes selected by ``fit_mask``."""
        ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform node features using fitted state."""
        ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Expose model-specific construction, loss, and scoring behavior."""

    name: str
    capabilities: ModelCapabilities
    parameter_defaults: Mapping[str, object]

    def validate_parameters(
        self,
        parameters: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Validate and normalize model-specific construction parameters."""
        ...

    def build(
        self,
        train_data: GraphData,
        parameters: Mapping[str, object],
    ) -> torch.nn.Module:
        """Construct the wrapped model without running optimization."""
        ...

    def compute_training_loss(
        self,
        model: torch.nn.Module,
        train_data: GraphData,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> TrainingLoss:
        """Compute one architecture-specific loss without stepping an optimizer."""
        ...

    def score_edges(
        self,
        model: torch.nn.Module,
        data: GraphData,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one existing-semantics score for each candidate edge."""
        ...

    def result_metadata(
        self,
        parameters: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Return model-specific fields retained with experiment results."""
        ...


@runtime_checkable
class EvaluationMetric(Protocol):
    """Compute one scalar metric from labels and prediction scores."""

    name: str
    input_kind: str

    def compute(self, labels: torch.Tensor, scores: torch.Tensor) -> float:
        """Compute the metric value."""
        ...


@runtime_checkable
class ResultStore(Protocol):
    """Persist evaluation results independently of a storage format."""

    def write(
        self,
        specification: ResolvedExperimentSpec,
        training: TrainingResult | None,
        evaluations: Sequence[EvaluationResult],
    ) -> None:
        """Persist one resolved experiment and its results."""
        ...


__all__ = [
    "DatasetProvider",
    "EvaluationMetric",
    "FeaturePreprocessor",
    "ModelAdapter",
    "ResultStore",
    "SplitProtocol",
]
