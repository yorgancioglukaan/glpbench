"""Canonical schema for validation-convergence audit configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glpbench.configuration.common import (
    DatasetConfig, ExecutionConfig, ModelConfig, OutputConfig, SplitConfig,
    TrainingConfig, parse_dataset_config, parse_execution_config,
    parse_model_config, parse_output_config, parse_split_config,
    parse_training_config,
)
from glpbench.configuration.parsing import (
    configuration_error, reject_unknown_keys, require_integer, require_keys,
    require_mapping, require_sequence, require_string,
)


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Store one strictly parsed validation-convergence audit."""

    schema_version: int
    kind: str
    datasets: DatasetConfig
    split: SplitConfig
    models: tuple[ModelConfig, ...]
    training: TrainingConfig
    execution: ExecutionConfig
    output: OutputConfig
    selected_configurations: Path | None = None


def parse_audit_config(document: Any) -> AuditConfig:
    """Parse a canonical validation-convergence audit document."""
    raw = require_mapping(document, ())
    allowed = {
        "schema_version", "kind", "datasets", "split", "models", "training",
        "execution", "output", "selected_configurations",
    }
    reject_unknown_keys(raw, allowed, ())
    require_keys(raw, allowed - {"selected_configurations"}, ())
    version = require_integer(raw["schema_version"], ("schema_version",), minimum=1)
    if version != 1:
        raise configuration_error(("schema_version",), "must be 1")
    kind = require_string(raw["kind"], ("kind",))
    if kind != "validation-convergence-audit":
        raise configuration_error(
            ("kind",), "must be 'validation-convergence-audit'"
        )
    models = tuple(
        parse_model_config(value, ("models", index))
        for index, value in enumerate(require_sequence(raw["models"], ("models",)))
    )
    if not models:
        raise configuration_error(("models",), "must not be empty")
    selected = None
    if "selected_configurations" in raw:
        selected = Path(require_string(
            raw["selected_configurations"], ("selected_configurations",)
        ))
    return AuditConfig(
        version, kind, parse_dataset_config(raw["datasets"]),
        parse_split_config(raw["split"]), models,
        parse_training_config(raw["training"]),
        parse_execution_config(raw["execution"]),
        parse_output_config(raw["output"]), selected,
    )


__all__ = ["AuditConfig", "parse_audit_config"]
