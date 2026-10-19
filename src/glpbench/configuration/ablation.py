"""Canonical schema for training-policy ablation configurations."""

from __future__ import annotations

from dataclasses import dataclass
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
class NamedTrainingConfig:
    """Associate a unique name with one training configuration."""

    name: str
    training: TrainingConfig


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Store one strictly parsed training-policy ablation."""

    schema_version: int
    kind: str
    datasets: DatasetConfig
    split: SplitConfig
    models: tuple[ModelConfig, ...]
    policies: tuple[NamedTrainingConfig, ...]
    execution: ExecutionConfig
    output: OutputConfig


def parse_ablation_config(document: Any) -> AblationConfig:
    """Parse a canonical training-policy ablation document."""
    raw = require_mapping(document, ())
    allowed = {
        "schema_version", "kind", "datasets", "split", "models", "policies",
        "execution", "output",
    }
    reject_unknown_keys(raw, allowed, ())
    require_keys(raw, allowed, ())
    version = require_integer(raw["schema_version"], ("schema_version",), minimum=1)
    if version != 1:
        raise configuration_error(("schema_version",), "must be 1")
    kind = require_string(raw["kind"], ("kind",))
    if kind != "training-policy-ablation":
        raise configuration_error(("kind",), "must be 'training-policy-ablation'")
    models = tuple(
        parse_model_config(value, ("models", index))
        for index, value in enumerate(require_sequence(raw["models"], ("models",)))
    )
    if not models:
        raise configuration_error(("models",), "must not be empty")
    policies = []
    for index, value in enumerate(require_sequence(raw["policies"], ("policies",))):
        path = ("policies", index)
        policy = require_mapping(value, path)
        reject_unknown_keys(policy, {"name", "training"}, path)
        require_keys(policy, {"name", "training"}, path)
        policies.append(NamedTrainingConfig(
            require_string(policy["name"], path + ("name",)),
            parse_training_config(policy["training"], path + ("training",)),
        ))
    if not policies:
        raise configuration_error(("policies",), "must not be empty")
    if len({policy.name for policy in policies}) != len(policies):
        raise configuration_error(("policies",), "must use unique names")
    return AblationConfig(
        version, kind, parse_dataset_config(raw["datasets"]),
        parse_split_config(raw["split"]), models, tuple(policies),
        parse_execution_config(raw["execution"]),
        parse_output_config(raw["output"]),
    )


__all__ = ["AblationConfig", "NamedTrainingConfig", "parse_ablation_config"]
