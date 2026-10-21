"""Structural interfaces for extensible GLPBench components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import torch

from glpbench.contracts import (
    AuditReport,
    EvaluationResult,
    GraphData,
    LoadedDataset,
    ModelCapabilities,
    ResolvedExperimentSpec,
    SplitBundle,
    TrainingPolicy,
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
class LinkPredictionModel(Protocol):
    """Train a link predictor and score candidate edges."""

    capabilities: ModelCapabilities

    def fit(
        self,
        train_data: GraphData,
        validation_data: GraphData | None,
        training_policy: TrainingPolicy,
    ) -> TrainingResult:
        """Fit the model according to a resolved training policy."""
        ...

    def score_edges(
        self,
        data: GraphData,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return one score for each candidate edge."""
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
    "LinkPredictionModel",
    "ResultStore",
    "SplitProtocol",
]
