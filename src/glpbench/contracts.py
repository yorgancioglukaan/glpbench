"""Immutable value contracts shared across GLPBench components.

The containers in this module are structurally immutable. PyG data objects and
PyTorch tensors stored inside them remain mutable and are never copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch_geometric.data import Data, HeteroData


GraphData = Data | HeteroData


def _require_non_empty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_optional_non_empty_string(value: str | None, name: str) -> None:
    if value is not None:
        _require_non_empty_string(value, name)


def _require_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_number(value: float, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any], name: str = "mapping") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _normalize_node_ids(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if any(
        not isinstance(node_id, int) or isinstance(node_id, bool) or node_id < 0
        for node_id in normalized
    ):
        raise ValueError(f"{name} must contain non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Describe the provenance and structural summary of one graph dataset."""

    name: str
    family: str
    num_nodes: int
    num_edges: int
    directed: bool
    has_node_features: bool
    num_node_features: int
    source_url: str | None = None
    citation: str | None = None
    dataset_version: str | None = None
    loader_preprocessing: tuple[str, ...] = ()
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "name")
        _require_non_empty_string(self.family, "family")
        _require_non_negative_integer(self.num_nodes, "num_nodes")
        _require_non_negative_integer(self.num_edges, "num_edges")
        _require_non_negative_integer(
            self.num_node_features, "num_node_features"
        )
        if not isinstance(self.directed, bool):
            raise TypeError("directed must be a boolean")
        if not isinstance(self.has_node_features, bool):
            raise TypeError("has_node_features must be a boolean")
        if not self.has_node_features and self.num_node_features != 0:
            raise ValueError(
                "num_node_features must be zero when node features are absent"
            )
        if self.has_node_features and self.num_node_features == 0:
            raise ValueError(
                "num_node_features must be positive when node features are present"
            )
        for value, name in (
            (self.source_url, "source_url"),
            (self.citation, "citation"),
            (self.dataset_version, "dataset_version"),
            (self.fingerprint, "fingerprint"),
        ):
            _require_optional_non_empty_string(value, name)
        preprocessing = tuple(self.loader_preprocessing)
        if any(not isinstance(item, str) or not item for item in preprocessing):
            raise ValueError(
                "loader_preprocessing must contain non-empty strings"
            )
        object.__setattr__(self, "loader_preprocessing", preprocessing)


@dataclass(frozen=True, slots=True, eq=False)
class LoadedDataset:
    """Pair native PyG graph data with GLPBench dataset metadata."""

    data: GraphData
    metadata: DatasetMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.data, (Data, HeteroData)):
            raise TypeError("data must be a PyG Data or HeteroData object")
        if not isinstance(self.metadata, DatasetMetadata):
            raise TypeError("metadata must be DatasetMetadata")
        if self.data.num_nodes is not None and self.data.num_nodes != self.metadata.num_nodes:
            raise ValueError("metadata num_nodes does not match data.num_nodes")
        if self.data.num_edges != self.metadata.num_edges:
            raise ValueError("metadata num_edges does not match data.num_edges")
        if isinstance(self.data, Data):
            features = getattr(self.data, "x", None)
            if self.metadata.has_node_features and features is None:
                raise ValueError("metadata declares node features but data.x is absent")
            if not self.metadata.has_node_features and features is not None:
                raise ValueError("metadata declares no node features but data.x is present")
            if features is not None and features.ndim != 2:
                raise ValueError("data.x must be two-dimensional")
            if features is not None and features.size(1) != self.metadata.num_node_features:
                raise ValueError(
                    "metadata num_node_features does not match data.x"
                )


@dataclass(frozen=True, slots=True)
class ProtocolMetadata:
    """Describe how a protocol generated a split bundle."""

    name: str
    schema_version: int
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requested_counts: Mapping[str, int] = field(default_factory=dict)
    realized_counts: Mapping[str, int] = field(default_factory=dict)
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "name")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("schema_version must be a positive integer")
        if self.seed is not None:
            _require_non_negative_integer(self.seed, "seed")
        _require_optional_non_empty_string(self.fingerprint, "fingerprint")
        object.__setattr__(
            self, "parameters", _freeze_mapping(self.parameters, "parameters")
        )
        for attribute in ("requested_counts", "realized_counts"):
            counts = getattr(self, attribute)
            for count_name, count in counts.items():
                _require_non_empty_string(count_name, f"{attribute} key")
                _require_non_negative_integer(count, f"{attribute}.{count_name}")
            object.__setattr__(self, attribute, _freeze_mapping(counts, attribute))


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Record the outcome of a split-protocol invariant audit."""

    passed: bool
    checks: Mapping[str, bool]
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        for check_name, outcome in self.checks.items():
            _require_non_empty_string(check_name, "check name")
            if not isinstance(outcome, bool):
                raise TypeError(f"check {check_name} outcome must be a boolean")
        for check_name, detail in self.details.items():
            _require_non_empty_string(check_name, "detail name")
            _require_non_empty_string(detail, f"detail {check_name}")
        if self.passed != all(self.checks.values()):
            raise ValueError("passed must agree with the individual check outcomes")
        object.__setattr__(self, "checks", _freeze_mapping(self.checks, "checks"))
        object.__setattr__(self, "details", _freeze_mapping(self.details, "details"))


@dataclass(frozen=True, slots=True, eq=False)
class SplitBundle:
    """Group native PyG partitions and metadata for one protocol realization."""

    train: GraphData
    validation: GraphData | None
    tests: Mapping[str, GraphData]
    metadata: ProtocolMetadata
    validation_query_nodes: tuple[int, ...] = ()
    test_query_nodes: tuple[int, ...] = ()
    visible_node_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.train, (Data, HeteroData)):
            raise TypeError("train must be a PyG Data or HeteroData object")
        if self.validation is not None and not isinstance(
            self.validation, (Data, HeteroData)
        ):
            raise TypeError("validation must be a PyG Data or HeteroData object")
        if not isinstance(self.metadata, ProtocolMetadata):
            raise TypeError("metadata must be ProtocolMetadata")
        for name, split in self.tests.items():
            _require_non_empty_string(name, "test split name")
            if not isinstance(split, (Data, HeteroData)):
                raise TypeError("test splits must be PyG Data or HeteroData objects")
        object.__setattr__(self, "tests", _freeze_mapping(self.tests, "tests"))
        validation_nodes = _normalize_node_ids(
            self.validation_query_nodes, "validation_query_nodes"
        )
        test_nodes = _normalize_node_ids(self.test_query_nodes, "test_query_nodes")
        if set(validation_nodes) & set(test_nodes):
            raise ValueError("validation and test query nodes must be disjoint")
        object.__setattr__(self, "validation_query_nodes", validation_nodes)
        object.__setattr__(self, "test_query_nodes", test_nodes)
        if self.visible_node_mask is not None:
            mask = self.visible_node_mask
            if not isinstance(mask, torch.Tensor):
                raise TypeError("visible_node_mask must be a torch.Tensor")
            if mask.dtype != torch.bool or mask.ndim != 1:
                raise ValueError("visible_node_mask must be a one-dimensional boolean tensor")
            if self.train.num_nodes is not None and mask.numel() != self.train.num_nodes:
                raise ValueError("visible_node_mask length must match train.num_nodes")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Declare the inputs and evaluation settings supported by a model."""

    trainable: bool = True
    uses_message_passing: bool = True
    uses_conv_layer: bool = True
    requires_graph_context: bool = False
    requires_reduced_features: bool = False
    supports_unseen_sources: bool = False
    supports_unseen_targets: bool = False
    requires_labels: bool = False
    requires_auxiliary_data: bool = False
    supports_agae_regularization: bool = False
    uses_mlp_decoder: bool = False
    output_kind: str = "raw-scores"
    supported_devices: tuple[str, ...] = ("cpu",)

    def __post_init__(self) -> None:
        boolean_fields = (
            "trainable",
            "uses_message_passing",
            "uses_conv_layer",
            "requires_graph_context",
            "requires_reduced_features",
            "supports_unseen_sources",
            "supports_unseen_targets",
            "requires_labels",
            "requires_auxiliary_data",
            "supports_agae_regularization",
            "uses_mlp_decoder",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.output_kind not in {"raw-scores", "probabilities", "rankings"}:
            raise ValueError(
                "output_kind must be raw-scores, probabilities, or rankings"
            )
        devices = tuple(self.supported_devices)
        if not devices or any(not isinstance(item, str) or not item for item in devices):
            raise ValueError("supported_devices must contain non-empty strings")
        if len(set(devices)) != len(devices):
            raise ValueError("supported_devices must not contain duplicates")
        object.__setattr__(self, "supported_devices", devices)


@dataclass(frozen=True, slots=True)
class TrainingPolicy:
    """Describe architecture-independent optimization and model selection."""

    max_epochs: int
    optimizer: str = "adam"
    optimizer_parameters: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_strategy: str = "best-validation"
    selection_metric: str = "roc_auc"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_minimum_epochs: int = 0
    scheduler: str = "none"
    scheduler_parameters: Mapping[str, Any] = field(default_factory=dict)
    deterministic: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_epochs, int)
            or isinstance(self.max_epochs, bool)
            or self.max_epochs <= 0
        ):
            raise ValueError("max_epochs must be a positive integer")
        for value, name in (
            (self.optimizer, "optimizer"),
            (self.checkpoint_strategy, "checkpoint_strategy"),
            (self.selection_metric, "selection_metric"),
            (self.scheduler, "scheduler"),
        ):
            _require_non_empty_string(value, name)
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be a positive integer")
        if (
            not isinstance(self.early_stopping_min_delta, (int, float))
            or isinstance(self.early_stopping_min_delta, bool)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be non-negative")
        if (
            not isinstance(self.early_stopping_minimum_epochs, int)
            or isinstance(self.early_stopping_minimum_epochs, bool)
            or not 0 <= self.early_stopping_minimum_epochs <= self.max_epochs
        ):
            raise ValueError(
                "early_stopping_minimum_epochs must be between zero and max_epochs"
            )
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        object.__setattr__(
            self,
            "optimizer_parameters",
            _freeze_mapping(self.optimizer_parameters, "optimizer_parameters"),
        )
        object.__setattr__(
            self,
            "scheduler_parameters",
            _freeze_mapping(self.scheduler_parameters, "scheduler_parameters"),
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Describe checkpoint selection, efficiency, and optimization outcomes."""

    epochs_executed: int
    final_epoch: int | None
    best_epoch: int | None
    selected_epoch: int | None
    checkpoint_strategy: str | None
    selection_metric: str | None
    best_validation_score: float | None
    stopped_early: bool
    training_time_seconds: float | None
    trainable_parameter_count: int
    selected_loss: float | None = None
    loss_history: tuple[float, ...] = ()
    final_learning_rate: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.epochs_executed, "epochs_executed")
        _require_non_negative_integer(
            self.trainable_parameter_count, "trainable_parameter_count"
        )
        for value, name in (
            (self.final_epoch, "final_epoch"),
            (self.best_epoch, "best_epoch"),
            (self.selected_epoch, "selected_epoch"),
        ):
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
                if value > self.epochs_executed:
                    raise ValueError(f"{name} cannot exceed epochs_executed")
        if self.epochs_executed == 0 and self.final_epoch is not None:
            raise ValueError("final_epoch must be absent when no epochs were executed")
        _require_optional_non_empty_string(
            self.checkpoint_strategy, "checkpoint_strategy"
        )
        _require_optional_non_empty_string(self.selection_metric, "selection_metric")
        if not isinstance(self.stopped_early, bool):
            raise TypeError("stopped_early must be a boolean")
        if self.best_validation_score is not None:
            _require_finite_number(
                self.best_validation_score, "best_validation_score"
            )
        if self.selected_loss is not None:
            _require_finite_number(self.selected_loss, "selected_loss")
        for value, name in (
            (self.training_time_seconds, "training_time_seconds"),
            (self.final_learning_rate, "final_learning_rate"),
        ):
            if value is not None:
                _require_finite_number(value, name)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        history = tuple(self.loss_history)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in history
        ):
            raise ValueError("loss_history must contain finite numbers")
        object.__setattr__(self, "loss_history", history)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Store metrics and candidate counts for one evaluation setting."""

    setting: str
    metrics: Mapping[str, float]
    positive_count: int
    negative_count: int
    evaluation_time_seconds: float | None = None
    run_id: str | None = None
    configuration_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.setting, "setting")
        _require_non_negative_integer(self.positive_count, "positive_count")
        _require_non_negative_integer(self.negative_count, "negative_count")
        for metric_name, value in self.metrics.items():
            _require_non_empty_string(metric_name, "metric name")
            _require_finite_number(value, f"metric {metric_name}")
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics, "metrics"))
        if self.evaluation_time_seconds is not None and (
            not isinstance(self.evaluation_time_seconds, (int, float))
            or isinstance(self.evaluation_time_seconds, bool)
            or not math.isfinite(self.evaluation_time_seconds)
            or self.evaluation_time_seconds < 0
        ):
            raise ValueError(
                "evaluation_time_seconds must be a finite non-negative number"
            )
        _require_optional_non_empty_string(self.run_id, "run_id")
        _require_optional_non_empty_string(self.configuration_id, "configuration_id")


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Pair prepared graph partitions with preprocessing runtime metadata."""

    splits: SplitBundle
    preprocessing_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.splits, SplitBundle):
            raise TypeError("splits must be a SplitBundle")
        if self.preprocessing_time_seconds is not None:
            _require_finite_number(
                self.preprocessing_time_seconds,
                "preprocessing_time_seconds",
            )
            if self.preprocessing_time_seconds < 0:
                raise ValueError("preprocessing_time_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class ResolvedExperimentSpec:
    """Describe one experiment after defaults and registrations are resolved."""

    schema_version: int
    dataset: str
    model_type: str
    protocol: str
    seed: int
    device: str
    training_policy: TrainingPolicy
    dataset_parameters: Mapping[str, Any] = field(default_factory=dict)
    protocol_parameters: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: str | None = None
    preprocessing_parameters: Mapping[str, Any] = field(default_factory=dict)
    model_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version <= 0
        ):
            raise ValueError("schema_version must be a positive integer")
        for value, name in (
            (self.dataset, "dataset"),
            (self.model_type, "model_type"),
            (self.protocol, "protocol"),
            (self.device, "device"),
        ):
            _require_non_empty_string(value, name)
        _require_non_negative_integer(self.seed, "seed")
        if not isinstance(self.training_policy, TrainingPolicy):
            raise TypeError("training_policy must be TrainingPolicy")
        _require_optional_non_empty_string(self.preprocessing, "preprocessing")
        for attribute in (
            "dataset_parameters",
            "protocol_parameters",
            "preprocessing_parameters",
            "model_parameters",
        ):
            object.__setattr__(
                self,
                attribute,
                _freeze_mapping(getattr(self, attribute), attribute),
            )


__all__ = [
    "AuditReport",
    "DatasetMetadata",
    "EvaluationResult",
    "LoadedDataset",
    "ModelCapabilities",
    "PreparationResult",
    "ProtocolMetadata",
    "ResolvedExperimentSpec",
    "SplitBundle",
    "TrainingPolicy",
    "TrainingResult",
]
