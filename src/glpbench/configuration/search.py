"""Canonical schema for hyperparameter-search configuration files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from glpbench.configuration.common import (
    DatasetConfig,
    ExecutionConfig,
    SplitConfig,
    TrainingConfig,
    parse_dataset_config,
    parse_execution_config,
    parse_split_config,
    parse_training_config,
)
from glpbench.configuration.common import _freeze_value
from glpbench.configuration.parsing import (
    configuration_error,
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


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Describe candidate values and their sampling strategy."""

    values: tuple[Any, ...]
    sampling: str = "random"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "values", tuple(_freeze_value(value) for value in self.values)
        )


@dataclass(frozen=True, slots=True)
class ModelSearchConfig:
    """Describe named search dimensions for one architecture."""

    architecture: str
    spaces: Mapping[str, SearchSpace] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spaces", MappingProxyType(dict(self.spaces)))


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Describe deterministic candidate generation."""

    strategy: str
    num_candidates: int
    candidate_seed: int
    include_default: bool


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Describe validation-based candidate selection."""

    metric: str
    aggregation: str
    validation_setting: str
    tie_tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchOutputConfig:
    """Describe search artifacts."""

    history: Path
    selected_configurations: Path
    failures: Path | None = None
    aggregates: Path | None = None


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Store one strictly parsed canonical search specification."""

    schema_version: int
    kind: str
    datasets: DatasetConfig
    split: SplitConfig
    models: tuple[ModelSearchConfig, ...]
    training: TrainingConfig
    execution: ExecutionConfig
    search: SearchSettings
    selection: SelectionConfig
    output: SearchOutputConfig


def _parse_space(value: Any, path: tuple[str | int, ...]) -> SearchSpace:
    raw = require_mapping(value, path)
    reject_unknown_keys(raw, {"values", "sampling"}, path)
    require_keys(raw, {"values"}, path)
    values = require_unique(
        require_sequence(raw["values"], path + ("values",)),
        path + ("values",),
    )
    sampling = require_string(raw.get("sampling", "random"), path + ("sampling",))
    if sampling not in {"random", "balanced", "stratified"}:
        raise configuration_error(path + ("sampling",), "is not supported")
    return SearchSpace(values=values, sampling=sampling)


def _parse_model_search(value: Any, path: tuple[str | int, ...]) -> ModelSearchConfig:
    raw = require_mapping(value, path)
    reject_unknown_keys(raw, {"architecture", "spaces"}, path)
    require_keys(raw, {"architecture", "spaces"}, path)
    spaces_raw = require_mapping(raw["spaces"], path + ("spaces",))
    spaces = {
        name: _parse_space(space, path + ("spaces", name))
        for name, space in spaces_raw.items()
    }
    return ModelSearchConfig(
        architecture=require_string(raw["architecture"], path + ("architecture",)),
        spaces=spaces,
    )


def _parse_search_settings(value: Any) -> SearchSettings:
    path = ("search",)
    raw = require_mapping(value, path)
    allowed = {"strategy", "num_candidates", "candidate_seed", "include_default"}
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, allowed, path)
    strategy = require_string(raw["strategy"], path + ("strategy",))
    if strategy != "seeded-random":
        raise configuration_error(path + ("strategy",), "must be 'seeded-random'")
    return SearchSettings(
        strategy=strategy,
        num_candidates=require_integer(
            raw["num_candidates"], path + ("num_candidates",), minimum=1
        ),
        candidate_seed=require_integer(
            raw["candidate_seed"], path + ("candidate_seed",), minimum=0
        ),
        include_default=require_boolean(
            raw["include_default"], path + ("include_default",)
        ),
    )


def _parse_selection(value: Any) -> SelectionConfig:
    path = ("selection",)
    raw = require_mapping(value, path)
    allowed = {"metric", "aggregation", "validation_setting", "tie_tolerance"}
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"metric", "aggregation", "validation_setting"}, path)
    return SelectionConfig(
        metric=require_string(raw["metric"], path + ("metric",)),
        aggregation=require_string(raw["aggregation"], path + ("aggregation",)),
        validation_setting=require_string(
            raw["validation_setting"], path + ("validation_setting",)
        ),
        tie_tolerance=require_number(
            raw.get("tie_tolerance", 0.0), path + ("tie_tolerance",), minimum=0
        ),
    )


def _parse_output(value: Any) -> SearchOutputConfig:
    path = ("output",)
    raw = require_mapping(value, path)
    allowed = {"history", "selected_configurations", "failures", "aggregates"}
    reject_unknown_keys(raw, allowed, path)
    require_keys(raw, {"history", "selected_configurations"}, path)
    parse_path = lambda key: Path(require_string(raw[key], path + (key,)))
    return SearchOutputConfig(
        history=parse_path("history"),
        selected_configurations=parse_path("selected_configurations"),
        failures=parse_path("failures") if "failures" in raw else None,
        aggregates=parse_path("aggregates") if "aggregates" in raw else None,
    )


def parse_search_config(document: Any) -> SearchConfig:
    """Parse a canonical hyperparameter-search document."""
    raw = require_mapping(document, ())
    allowed = {
        "schema_version", "kind", "datasets", "split", "models",
        "training", "execution", "search", "selection", "output",
    }
    reject_unknown_keys(raw, allowed, ())
    require_keys(raw, allowed, ())
    version = require_integer(raw["schema_version"], ("schema_version",), minimum=1)
    if version != 1:
        raise configuration_error(("schema_version",), "must be 1")
    kind = require_string(raw["kind"], ("kind",))
    if kind != "hyperparameter-search":
        raise configuration_error(("kind",), "must be 'hyperparameter-search'")
    model_values = require_sequence(raw["models"], ("models",))
    models = tuple(
        _parse_model_search(value, ("models", index))
        for index, value in enumerate(model_values)
    )
    if not models:
        raise configuration_error(("models",), "must not be empty")
    require_unique(tuple(model.architecture for model in models), ("models",))
    return SearchConfig(
        version, kind, parse_dataset_config(raw["datasets"]),
        parse_split_config(raw["split"]), models,
        parse_training_config(raw["training"]),
        parse_execution_config(raw["execution"]),
        _parse_search_settings(raw["search"]),
        _parse_selection(raw["selection"]), _parse_output(raw["output"]),
    )


__all__ = [
    "ModelSearchConfig", "SearchConfig", "SearchOutputConfig", "SearchSettings",
    "SearchSpace", "SelectionConfig", "parse_search_config",
]
