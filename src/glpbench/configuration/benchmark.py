"""Canonical schema for benchmark-grid configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from glpbench.configuration.common import (
    DatasetConfig,
    ExecutionConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
    TrainingConfig,
    parse_dataset_config,
    parse_execution_config,
    parse_model_config,
    parse_output_config,
    parse_split_config,
    parse_training_config,
)
from glpbench.configuration.parsing import (
    configuration_error,
    reject_unknown_keys,
    require_integer,
    require_keys,
    require_mapping,
    require_sequence,
    require_string,
    require_unique,
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Store one strictly parsed canonical benchmark specification."""

    schema_version: int
    kind: str
    datasets: DatasetConfig
    split: SplitConfig
    models: tuple[ModelConfig, ...]
    training: TrainingConfig
    execution: ExecutionConfig
    output: OutputConfig


def parse_benchmark_config(document: Any) -> BenchmarkConfig:
    """Parse a canonical benchmark document into an immutable contract."""
    raw = require_mapping(document, ())
    allowed = {
        "schema_version", "kind", "datasets", "split", "models",
        "training", "execution", "output",
    }
    reject_unknown_keys(raw, allowed, ())
    require_keys(raw, allowed, ())
    version = require_integer(raw["schema_version"], ("schema_version",), minimum=1)
    if version != 1:
        raise configuration_error(("schema_version",), "must be 1")
    kind = require_string(raw["kind"], ("kind",))
    if kind != "benchmark":
        raise configuration_error(("kind",), "must be 'benchmark'")
    model_values = require_sequence(raw["models"], ("models",))
    models = tuple(
        parse_model_config(model, ("models", index))
        for index, model in enumerate(model_values)
    )
    if not models:
        raise configuration_error(("models",), "must not be empty")
    require_unique(tuple(model.architecture for model in models), ("models",))
    return BenchmarkConfig(
        schema_version=version,
        kind=kind,
        datasets=parse_dataset_config(raw["datasets"]),
        split=parse_split_config(raw["split"]),
        models=models,
        training=parse_training_config(raw["training"]),
        execution=parse_execution_config(raw["execution"]),
        output=parse_output_config(raw["output"]),
    )


__all__ = ["BenchmarkConfig", "parse_benchmark_config"]
