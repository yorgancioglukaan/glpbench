"""Shared immutable sections for canonical GLPBench configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from glpbench.configuration.parsing import (
    FieldPath,
    optional,
    reject_unknown_keys,
    require_boolean,
    require_integer,
    require_keys,
    require_mapping,
    require_number,
    require_sequence,
    require_string,
    require_unique,
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = {}
    for key, item in value.items():
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Select datasets and their loader-level preprocessing."""

    names: tuple[str, ...]
    root: Path = Path("data")
    normalize_features: bool = True


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Describe a graph splitting protocol and its query sizes."""

    protocol: str
    train_ratio: float
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Describe one architecture and its fixed or grid parameters."""

    architecture: str
    conv_layer_types: tuple[str, ...] = ()
    reduction_methods: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    """Describe optional validation-based early stopping."""

    enabled: bool = False
    patience: int | None = None
    min_delta: float = 0.0
    minimum_epochs: int = 0


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Describe a learning-rate scheduler and its parameters."""

    type: str = "none"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Describe optimization, checkpointing, and stopping settings."""

    max_epochs: int
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    checkpoint_strategy: str = "best-validation"
    selection_metric: str = "roc_auc"
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Describe reproducibility and execution settings."""

    seeds: tuple[int, ...]
    device: str = "cpu"
    verbose: bool = False
    fail_fast: bool = False


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Describe result and optional auxiliary artifact paths."""

    results: Path
    failures: Path | None = None
    epoch_history: Path | None = None


def _parse_string_sequence(value: Any, path: FieldPath) -> tuple[str, ...]:
    values = require_sequence(value, path)
    parsed = tuple(
        require_string(item, path + (index,))
        for index, item in enumerate(values)
    )
    return require_unique(parsed, path)


def _parse_path(value: Any, path: FieldPath) -> Path:
    return Path(require_string(value, path))


def parse_dataset_config(value: Any, path: FieldPath = ("datasets",)) -> DatasetConfig:
    """Parse a strict dataset section."""
    raw = require_mapping(value, path)
    allowed = {"names", "root", "normalize_features"}
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"names"}, path)
    return DatasetConfig(
        names=_parse_string_sequence(raw["names"], path + ("names",)),
        root=optional(raw, "root", _parse_path, path, Path("data")),
        normalize_features=optional(
            raw, "normalize_features", require_boolean, path, True
        ),
    )


def parse_split_config(value: Any, path: FieldPath = ("split",)) -> SplitConfig:
    """Parse a strict split-protocol section."""
    raw = require_mapping(value, path)
    allowed = {
        "protocol", "train_ratio", "query_size", "query_fraction",
        "validation_query_size", "validation_query_fraction",
    }
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"protocol", "train_ratio"}, path)
    query_size = optional(
        raw, "query_size",
        lambda item, item_path: require_integer(item, item_path, minimum=1),
        path,
    )
    query_fraction = optional(
        raw, "query_fraction",
        lambda item, item_path: require_number(
            item, item_path, minimum=0, maximum=1,
            minimum_exclusive=True, maximum_exclusive=True,
        ),
        path,
    )
    validation_size = optional(
        raw, "validation_query_size",
        lambda item, item_path: require_integer(item, item_path, minimum=1),
        path,
    )
    validation_fraction = optional(
        raw, "validation_query_fraction",
        lambda item, item_path: require_number(
            item, item_path, minimum=0, maximum=1,
            minimum_exclusive=True, maximum_exclusive=True,
        ),
        path,
    )
    from glpbench.configuration.parsing import configuration_error
    if (query_size is None) == (query_fraction is None):
        raise configuration_error(
            path, "must define exactly one of query_size and query_fraction"
        )
    if validation_size is not None and validation_fraction is not None:
        raise configuration_error(
            path,
            "cannot define both validation_query_size and validation_query_fraction",
        )
    return SplitConfig(
        protocol=require_string(raw["protocol"], path + ("protocol",)),
        train_ratio=require_number(
            raw["train_ratio"], path + ("train_ratio",), minimum=0,
            maximum=1, minimum_exclusive=True, maximum_exclusive=True,
        ),
        query_size=query_size,
        query_fraction=query_fraction,
        validation_query_size=validation_size,
        validation_query_fraction=validation_fraction,
    )


def parse_model_config(value: Any, path: FieldPath) -> ModelConfig:
    """Parse one strict model section."""
    raw = require_mapping(value, path)
    allowed = {
        "architecture", "conv_layer_types", "reduction_methods", "parameters"
    }
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"architecture"}, path)
    parameters = optional(raw, "parameters", require_mapping, path, {})
    return ModelConfig(
        architecture=require_string(
            raw["architecture"], path + ("architecture",)
        ),
        conv_layer_types=(
            _parse_string_sequence(
                raw["conv_layer_types"], path + ("conv_layer_types",)
            ) if "conv_layer_types" in raw else ()
        ),
        reduction_methods=(
            _parse_string_sequence(
                raw["reduction_methods"], path + ("reduction_methods",)
            ) if "reduction_methods" in raw else ()
        ),
        parameters=parameters,
    )


def parse_early_stopping_config(
    value: Any, path: FieldPath
) -> EarlyStoppingConfig:
    """Parse early-stopping settings with cross-field validation."""
    raw = require_mapping(value, path)
    allowed = {"enabled", "patience", "min_delta", "minimum_epochs"}
    reject_unknown_keys(raw, allowed, path)
    enabled = optional(raw, "enabled", require_boolean, path, False)
    patience = optional(
        raw, "patience",
        lambda item, item_path: require_integer(item, item_path, minimum=1),
        path,
    )
    from glpbench.configuration.parsing import configuration_error
    if enabled and patience is None:
        raise configuration_error(path + ("patience",), "is required when enabled")
    if not enabled and patience is not None:
        raise configuration_error(path + ("patience",), "requires enabled to be true")
    return EarlyStoppingConfig(
        enabled=enabled,
        patience=patience,
        min_delta=optional(
            raw, "min_delta",
            lambda item, item_path: require_number(item, item_path, minimum=0),
            path, 0.0,
        ),
        minimum_epochs=optional(
            raw, "minimum_epochs",
            lambda item, item_path: require_integer(item, item_path, minimum=0),
            path, 0,
        ),
    )


def parse_scheduler_config(value: Any, path: FieldPath) -> SchedulerConfig:
    """Parse a scheduler section without interpreting scheduler parameters."""
    raw = require_mapping(value, path)
    reject_unknown_keys(raw, {"type", "parameters"}, path)
    scheduler_type = optional(raw, "type", require_string, path, "none")
    parameters = optional(raw, "parameters", require_mapping, path, {})
    return SchedulerConfig(scheduler_type, parameters)


def parse_training_config(
    value: Any, path: FieldPath = ("training",)
) -> TrainingConfig:
    """Parse strict common training settings."""
    raw = require_mapping(value, path)
    allowed = {
        "max_epochs", "learning_rate", "weight_decay", "checkpoint_strategy",
        "selection_metric", "early_stopping", "scheduler",
    }
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"max_epochs"}, path)
    stopping = (
        parse_early_stopping_config(
            raw["early_stopping"], path + ("early_stopping",)
        ) if "early_stopping" in raw else EarlyStoppingConfig()
    )
    max_epochs = require_integer(
        raw["max_epochs"], path + ("max_epochs",), minimum=1
    )
    if stopping.minimum_epochs > max_epochs:
        from glpbench.configuration.parsing import configuration_error
        raise configuration_error(
            path + ("early_stopping", "minimum_epochs"),
            "cannot exceed training.max_epochs",
        )
    return TrainingConfig(
        max_epochs=max_epochs,
        learning_rate=optional(
            raw, "learning_rate",
            lambda item, item_path: require_number(
                item, item_path, minimum=0, minimum_exclusive=True
            ), path, 0.01,
        ),
        weight_decay=optional(
            raw, "weight_decay",
            lambda item, item_path: require_number(item, item_path, minimum=0),
            path, 0.0,
        ),
        checkpoint_strategy=optional(
            raw, "checkpoint_strategy", require_string, path, "best-validation"
        ),
        selection_metric=optional(
            raw, "selection_metric", require_string, path, "roc_auc"
        ),
        early_stopping=stopping,
        scheduler=(
            parse_scheduler_config(raw["scheduler"], path + ("scheduler",))
            if "scheduler" in raw else SchedulerConfig()
        ),
    )


def parse_execution_config(
    value: Any, path: FieldPath = ("execution",)
) -> ExecutionConfig:
    """Parse strict execution settings."""
    raw = require_mapping(value, path)
    allowed = {"seeds", "device", "verbose", "fail_fast"}
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"seeds"}, path)
    values = require_sequence(raw["seeds"], path + ("seeds",))
    seeds = tuple(
        require_integer(seed, path + ("seeds", index), minimum=0)
        for index, seed in enumerate(values)
    )
    return ExecutionConfig(
        seeds=require_unique(seeds, path + ("seeds",)),
        device=optional(raw, "device", require_string, path, "cpu"),
        verbose=optional(raw, "verbose", require_boolean, path, False),
        fail_fast=optional(raw, "fail_fast", require_boolean, path, False),
    )


def parse_output_config(value: Any, path: FieldPath = ("output",)) -> OutputConfig:
    """Parse strict output-artifact settings."""
    raw = require_mapping(value, path)
    allowed = {"results", "failures", "epoch_history"}
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"results"}, path)
    return OutputConfig(
        results=_parse_path(raw["results"], path + ("results",)),
        failures=optional(raw, "failures", _parse_path, path),
        epoch_history=optional(raw, "epoch_history", _parse_path, path),
    )


__all__ = [
    "DatasetConfig", "EarlyStoppingConfig", "ExecutionConfig", "ModelConfig",
    "OutputConfig", "SchedulerConfig", "SplitConfig", "TrainingConfig",
    "parse_dataset_config", "parse_execution_config", "parse_model_config",
    "parse_output_config", "parse_split_config", "parse_training_config",
]
