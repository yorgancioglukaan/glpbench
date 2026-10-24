"""Run configuration-driven link-prediction benchmark grids."""

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from tqdm.auto import tqdm
import yaml

from glpbench.exceptions import ConfigurationError
from glpbench.datasets import available_datasets
from glpbench.logging import (
    CSVEpochHistoryLogger,
    CSVResultLogger,
    EpochHistoryRecord,
    make_configuration_id,
)
from glpbench.preprocessing import available_reduction_preprocessors
from glpbench.models import (
    available_agae_regularizers,
    available_model_types,
    get_model_adapter,
    resolve_model_reference,
)
from glpbench.training import validate_scheduler_config
from glpbench.runners.experiment import run_experiment


_CONV_LAYER_TYPES = ("GCNConv", "GraphConv", "SAGEConv", "GATv2Conv")
_DEFAULTS = {
    "query_size": 10,
    "query_fraction": None,
    "validation_query_size": None,
    "validation_query_fraction": None,
    "train_ratio": 0.8,
    "epochs": 100,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "hidden_channels": 128,
    "out_channels": 64,
    "conv_layer_type": "GCNConv",
    "reduction_method": "pca",
    "decoder_hidden_channels": None,
    "decoder_dropout": 0.0,
    "kl_weight": 1.0,
    "selection_metric": "roc_auc",
    "early_stopping_patience": None,
    "early_stopping_min_delta": 0.0,
    "early_stopping_minimum_epochs": 0,
    "checkpoint_strategy": "best-validation",
    "training_policy": "default",
    "lr_scheduler": "none",
    "lr_scheduler_kwargs": None,
    "normalize_features": True,
    "device": "cpu",
    "gatv2_kwargs": None,
    "deal_theta_ss": 1.0,
    "deal_theta_aa": 1.0,
    "deal_theta_sa": 1.0,
    "deal_lambda_ss": 1.0,
    "deal_lambda_aa": 1.0,
    "deal_lambda_sa": 1.0,
    "deal_similarity_scale": 1.0,
    "deal_similarity_margin": 0.0,
}
@dataclass(frozen=True)
class RegularizationConfig:
    """Describe one AGAE regularization setting."""

    type: str
    strength: float = 0.0


@dataclass(frozen=True)
class ModelGrid:
    """Describe model-specific dimensions of a benchmark grid."""

    type: str
    conv_layer_types: tuple[str, ...]
    reduction_methods: tuple[str, ...]
    regularization: tuple[RegularizationConfig, ...]
    parameters: dict


@dataclass(frozen=True)
class TrainingPolicy:
    """Describe one checkpoint-selection and stopping policy."""

    name: str
    max_epochs: int
    checkpoint_strategy: str
    early_stopping_patience: int | None
    early_stopping_min_delta: float
    early_stopping_minimum_epochs: int
    lr_scheduler: str
    lr_scheduler_kwargs: dict


@dataclass(frozen=True)
class BenchmarkConfig:
    """Describe a validated benchmark configuration file."""

    output: str
    data_root: str
    seeds: tuple[int, ...]
    defaults: dict
    datasets: tuple[str, ...]
    models: tuple[ModelGrid, ...]
    error_output: str
    selected_configurations: Path | None = None
    kind: str = "benchmark"
    training_policies: tuple[TrainingPolicy, ...] = ()
    epoch_history_output: str | None = None
    log_epoch_history: bool = False


@dataclass(frozen=True)
class ExperimentJob:
    """Store one fully resolved call to ``main.run_experiment``."""

    dataset: str
    model_type: str
    conv_layer_type: str
    reduction_method: str
    regularization: RegularizationConfig
    seeds: tuple[int, ...]
    parameters: dict

    def to_namespace(self, *, verbose=False):
        """Return arguments compatible with ``main.run_experiment``."""
        values = dict(self.parameters)
        values.update({
            "dataset": self.dataset,
            "model_type": [self.model_type],
            "conv_layer_type": self.conv_layer_type,
            "reduction_method": self.reduction_method,
            "agae_regularization": [self.regularization.type],
            "regularization_strength": self.regularization.strength,
            "seed": list(self.seeds),
            "num_seeds": None,
            "verbose": verbose,
        })
        return SimpleNamespace(**values)

    @property
    def label(self):
        """Return a compact human-readable job label."""
        parts = [self.dataset, self.model_type]
        if get_model_adapter(self.model_type).capabilities.uses_conv_layer:
            parts.append(self.conv_layer_type)
        if get_model_adapter(self.model_type).capabilities.requires_reduced_features:
            parts.extend((self.reduction_method, self.regularization.type))
        policy = self.parameters.get("training_policy", "default")
        if policy != "default":
            parts.append(policy)
        return "/".join(parts)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Summarize one benchmark-runner invocation."""

    run_id: str
    jobs_total: int
    jobs_succeeded: int
    jobs_failed: int
    expected_result_rows: int


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_unique_sequence(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _resolve_dataset_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset names must be non-empty strings")
    aliases = {
        "".join(character for character in candidate if character.isalnum()): candidate
        for candidate in available_datasets()
    }
    compact = "".join(
        character for character in name.lower() if character.isalnum()
    )
    if compact not in aliases:
        raise ValueError(f"Unknown dataset '{name}'")
    return aliases[compact]


def _validate_defaults(defaults):
    defaults = _require_mapping(defaults, "defaults")
    unknown = set(defaults).difference(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown default setting: {sorted(unknown)[0]}")
    resolved = {**_DEFAULTS, **defaults}
    if "query_fraction" in defaults and "query_size" not in defaults:
        resolved["query_size"] = None
    if (
        "validation_query_fraction" in defaults
        and "validation_query_size" not in defaults
    ):
        resolved["validation_query_size"] = None
    if resolved["query_size"] is not None and resolved["query_fraction"] is not None:
        raise ValueError("defaults cannot specify both query_size and query_fraction")
    if (
        resolved["validation_query_size"] is not None
        and resolved["validation_query_fraction"] is not None
    ):
        raise ValueError(
            "defaults cannot specify both validation query size and fraction"
        )
    if resolved["conv_layer_type"] not in _CONV_LAYER_TYPES:
        raise ValueError("Unknown default convolution layer")
    if resolved["reduction_method"] not in available_reduction_preprocessors():
        raise ValueError("Unknown default reduction method")
    if resolved["selection_metric"] not in {"roc_auc", "average_precision"}:
        raise ValueError("Unknown selection metric")
    if resolved["checkpoint_strategy"] not in {"best-validation", "final"}:
        raise ValueError("Unknown checkpoint strategy")
    if (
        not isinstance(resolved["training_policy"], str)
        or not resolved["training_policy"].strip()
    ):
        raise ValueError("training_policy must be a non-empty string")
    validate_scheduler_config(
        resolved["lr_scheduler"],
        resolved["lr_scheduler_kwargs"],
        epochs=resolved["epochs"],
    )
    if not isinstance(resolved["normalize_features"], bool):
        raise ValueError("normalize_features must be boolean")
    if not isinstance(resolved["epochs"], int) or resolved["epochs"] <= 0:
        raise ValueError("epochs must be a positive integer")
    if not 0 < resolved["train_ratio"] < 1:
        raise ValueError("train_ratio must be between zero and one")
    return resolved


def _parse_regularization(raw, model_type):
    if raw is None:
        return (RegularizationConfig("none", 0.0),)
    if not get_model_adapter(model_type).capabilities.supports_agae_regularization:
        raise ValueError("regularization is only supported by asymmetric models")
    entries = _require_unique_sequence(raw, "regularization")
    parsed = []
    for index, entry in enumerate(entries):
        entry = _require_mapping(entry, f"regularization[{index}]")
        unknown = set(entry).difference({"type", "strength"})
        if unknown:
            raise ValueError(f"Unknown regularization setting: {sorted(unknown)[0]}")
        kind = entry.get("type")
        strength = entry.get("strength", 0.0)
        if kind not in available_agae_regularizers():
            raise ValueError(f"Unknown AGAE regularizer '{kind}'")
        if not isinstance(strength, (int, float)) or strength < 0:
            raise ValueError("regularization strength must be non-negative")
        parsed.append(RegularizationConfig(kind, float(strength)))
    return tuple(parsed)


def _parse_model_grid(raw, defaults, index):
    raw = _require_mapping(raw, f"models[{index}]")
    unknown = set(raw).difference({
        "type", "conv_layer_types", "reduction_methods", "regularization",
        "parameters",
    })
    if unknown:
        raise ValueError(f"Unknown model-grid setting: {sorted(unknown)[0]}")
    model_type = raw.get("type")
    if model_type not in available_model_types():
        raise ValueError(f"Unknown model type '{model_type}'")
    model_adapter = get_model_adapter(model_type)

    raw_conv = raw.get("conv_layer_types")
    if not model_adapter.capabilities.uses_conv_layer:
        if raw_conv is not None:
            raise ValueError(f"{model_type} does not accept conv_layer_types")
        conv_layers = (defaults["conv_layer_type"],)
    else:
        raw_conv = (
            [defaults["conv_layer_type"]] if raw_conv is None
            else _require_unique_sequence(raw_conv, "conv_layer_types")
        )
        if any(layer not in _CONV_LAYER_TYPES for layer in raw_conv):
            raise ValueError("Unknown convolution layer type")
        conv_layers = tuple(raw_conv)

    raw_reducers = raw.get("reduction_methods")
    if not model_adapter.capabilities.requires_reduced_features and raw_reducers is not None:
        raise ValueError("reduction_methods are only supported by asymmetric")
    raw_reducers = (
        [defaults["reduction_method"]] if raw_reducers is None
        else _require_unique_sequence(raw_reducers, "reduction_methods")
    )
    if any(
        method not in available_reduction_preprocessors()
        for method in raw_reducers
    ):
        raise ValueError("Unknown reduction method")

    parameters = _require_mapping(raw.get("parameters", {}), "parameters")
    aliases = {
        "deal_theta_ss": "theta_ss",
        "deal_theta_aa": "theta_aa",
        "deal_theta_sa": "theta_sa",
        "deal_lambda_ss": "lambda_ss",
        "deal_lambda_aa": "lambda_aa",
        "deal_lambda_sa": "lambda_sa",
        "deal_similarity_scale": "similarity_scale",
        "deal_similarity_margin": "similarity_margin",
    }
    declared = set(model_adapter.parameter_defaults)
    training_parameters = {"learning_rate", "weight_decay"}
    unknown_parameters = {
        name for name in parameters
        if aliases.get(name, name) not in declared
        and name not in training_parameters
    }
    if unknown_parameters:
        raise ValueError(
            f"Unknown model parameter: {sorted(unknown_parameters)[0]}"
        )
    if "gatv2_kwargs" in parameters and "GATv2Conv" not in conv_layers:
        raise ValueError("gatv2_kwargs require GATv2Conv")
    declared_values = {
        aliases.get(name, name): value
        for name, value in parameters.items()
        if aliases.get(name, name) in declared
    }
    try:
        model_adapter.validate_parameters(declared_values)
    except (ConfigurationError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid parameters for {model_adapter.name}: {error}"
        ) from error
    return ModelGrid(
        type=model_type,
        conv_layer_types=conv_layers,
        reduction_methods=tuple(raw_reducers),
        regularization=_parse_regularization(
            raw.get("regularization"), model_type
        ),
        parameters=dict(parameters),
    )


def _validate_job_parameters(parameters):
    """Validate resolved values before any benchmark job executes."""
    for name in ("hidden_channels", "out_channels", "epochs"):
        value = parameters[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("query_size", "validation_query_size"):
        value = parameters[name]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("query_fraction", "validation_query_fraction"):
        value = parameters[name]
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name} must be between zero and one")
    if not 0 < parameters["train_ratio"] < 1:
        raise ValueError("train_ratio must be between zero and one")
    if parameters["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if parameters["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= parameters["decoder_dropout"] <= 1:
        raise ValueError("decoder_dropout must be between zero and one")
    decoder_hidden = parameters["decoder_hidden_channels"]
    if decoder_hidden is not None and (
        not isinstance(decoder_hidden, int)
        or isinstance(decoder_hidden, bool)
        or decoder_hidden <= 0
    ):
        raise ValueError("decoder_hidden_channels must be positive")
    if parameters["kl_weight"] < 0:
        raise ValueError("kl_weight must be non-negative")
    patience = parameters["early_stopping_patience"]
    if patience is not None and (
        not isinstance(patience, int)
        or isinstance(patience, bool)
        or patience <= 0
    ):
        raise ValueError("early_stopping_patience must be positive")
    if parameters["early_stopping_min_delta"] < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    minimum_epochs = parameters["early_stopping_minimum_epochs"]
    if (
        not isinstance(minimum_epochs, int)
        or isinstance(minimum_epochs, bool)
        or not 0 <= minimum_epochs <= parameters["epochs"]
    ):
        raise ValueError(
            "early_stopping_minimum_epochs must be between zero and epochs"
        )
    if parameters["checkpoint_strategy"] not in {"best-validation", "final"}:
        raise ValueError("Unknown checkpoint strategy")
    if (
        not isinstance(parameters["training_policy"], str)
        or not parameters["training_policy"].strip()
    ):
        raise ValueError("training_policy must be a non-empty string")
    validate_scheduler_config(
        parameters["lr_scheduler"],
        parameters["lr_scheduler_kwargs"],
        epochs=parameters["epochs"],
    )
    if not isinstance(parameters["device"], str) or not parameters["device"]:
        raise ValueError("device must be a non-empty string")
    gatv2_kwargs = parameters["gatv2_kwargs"]
    if gatv2_kwargs is not None and not isinstance(gatv2_kwargs, dict):
        raise ValueError("gatv2_kwargs must be a mapping")


def _selected_configuration_path(raw):
    """Validate a selected-hyperparameter artifact reference."""
    selected_raw = _require_mapping(raw, "selected_configurations")
    unknown = set(selected_raw).difference({
        "path", "granularity", "require_all_pairs"
    })
    if unknown:
        raise ValueError(
            "Unknown selected-configuration setting: "
            f"{sorted(unknown)[0]}"
        )
    if selected_raw.get("granularity") != ["dataset", "architecture"]:
        raise ValueError(
            "selected configuration granularity must be dataset and architecture"
        )
    if selected_raw.get("require_all_pairs") is not True:
        raise ValueError("selected configurations must require all pairs")
    selected_value = selected_raw.get("path")
    if not isinstance(selected_value, str) or not selected_value.strip():
        raise ValueError("selected_configurations.path must be non-empty")
    return Path(selected_value)


def _parse_training_policy(raw, index):
    """Validate one training-policy ablation entry."""
    raw = _require_mapping(raw, f"training_policies[{index}]")
    unknown = set(raw).difference({
        "name", "max_epochs", "checkpoint", "early_stopping", "scheduler"
    })
    if unknown:
        raise ValueError(f"Unknown training-policy setting: {sorted(unknown)[0]}")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("training policy name must be non-empty")
    max_epochs = raw.get("max_epochs")
    if (
        not isinstance(max_epochs, int)
        or isinstance(max_epochs, bool)
        or max_epochs <= 0
    ):
        raise ValueError("training policy max_epochs must be positive")
    checkpoint = raw.get("checkpoint")
    if checkpoint not in {"best-validation", "final"}:
        raise ValueError("Unknown training-policy checkpoint strategy")

    stopping = _require_mapping(
        raw.get("early_stopping"),
        f"training_policies[{index}].early_stopping",
    )
    unknown_stopping = set(stopping).difference({
        "enabled", "patience", "min_delta", "minimum_epochs"
    })
    if unknown_stopping:
        raise ValueError(
            "Unknown early-stopping setting: "
            f"{sorted(unknown_stopping)[0]}"
        )
    enabled = stopping.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("early_stopping.enabled must be boolean")
    patience = stopping.get("patience") if enabled else None
    if enabled and (
        not isinstance(patience, int)
        or isinstance(patience, bool)
        or patience <= 0
    ):
        raise ValueError("enabled early stopping requires positive patience")
    min_delta = stopping.get("min_delta", 0.0)
    minimum_epochs = stopping.get("minimum_epochs", 0)
    if (
        not isinstance(min_delta, (int, float))
        or isinstance(min_delta, bool)
        or min_delta < 0
    ):
        raise ValueError("early-stopping min_delta must be non-negative")
    if (
        not isinstance(minimum_epochs, int)
        or isinstance(minimum_epochs, bool)
        or not 0 <= minimum_epochs <= max_epochs
    ):
        raise ValueError(
            "early-stopping minimum_epochs must be between zero and max_epochs"
        )
    if not enabled and set(stopping).difference({"enabled"}):
        raise ValueError(
            "disabled early stopping cannot define patience, min_delta, or "
            "minimum_epochs"
        )
    scheduler = _require_mapping(
        raw.get("scheduler", {"type": "none"}),
        f"training_policies[{index}].scheduler",
    )
    scheduler_type = scheduler.get("type")
    scheduler_parameters = {
        name: value for name, value in scheduler.items() if name != "type"
    }
    scheduler_parameters = validate_scheduler_config(
        scheduler_type, scheduler_parameters, epochs=max_epochs
    )
    return TrainingPolicy(
        name=name,
        max_epochs=max_epochs,
        checkpoint_strategy=checkpoint,
        early_stopping_patience=patience,
        early_stopping_min_delta=float(min_delta),
        early_stopping_minimum_epochs=minimum_epochs,
        lr_scheduler=scheduler_type,
        lr_scheduler_kwargs=scheduler_parameters,
    )


def _load_training_policy_ablation(raw):
    """Validate a training-policy ablation configuration."""
    unknown = set(raw).difference({
        "version", "kind", "output", "error_output", "epoch_history_output",
        "data_root", "seeds", "selected_configurations", "datasets", "models",
        "splits", "evaluation", "training", "training_policies",
    })
    if unknown:
        raise ValueError(f"Unknown ablation setting: {sorted(unknown)[0]}")
    if raw.get("version") != 1:
        raise ValueError("training-policy ablations require version 1")

    output = raw.get("output", "out/training_policy_ablation.csv")
    error_output = raw.get(
        "error_output", str(Path(output).with_suffix(".errors.jsonl"))
    )
    epoch_output = raw.get("epoch_history_output")
    data_root = raw.get("data_root", "data")
    for value, name in (
        (output, "output"), (error_output, "error_output"),
        (epoch_output, "epoch_history_output"), (data_root, "data_root"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    seeds = _require_unique_sequence(raw.get("seeds"), "seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be non-negative integers")
    datasets = tuple(
        _resolve_dataset_name(name)
        for name in _require_unique_sequence(raw.get("datasets"), "datasets")
    )

    splits = _require_mapping(raw.get("splits"), "splits")
    if set(splits) != {
        "protocol", "query_fraction", "validation_query_fraction",
        "train_ratio", "paired_across_policies",
    }:
        raise ValueError("splits must define the training-ablation protocol")
    if splits["protocol"] != "matched-semi-inductive":
        raise ValueError("unsupported ablation split protocol")
    if splits["paired_across_policies"] is not True:
        raise ValueError("ablation splits must be paired across policies")

    evaluation = _require_mapping(raw.get("evaluation"), "evaluation")
    if set(evaluation).difference({
        "split_types", "primary_split", "primary_metric", "secondary_metrics"
    }):
        raise ValueError("Unknown evaluation setting")
    if evaluation.get("split_types") != ["transductive", "inductive"]:
        raise ValueError("ablation must evaluate both split types")
    if evaluation.get("primary_split") != "inductive":
        raise ValueError("ablation primary split must be inductive")
    if evaluation.get("primary_metric") != "roc_auc":
        raise ValueError("ablation primary metric must be roc_auc")
    if evaluation.get("secondary_metrics") != ["average_precision"]:
        raise ValueError("ablation secondary metric must be average_precision")

    training = _require_mapping(raw.get("training"), "training")
    unknown_training = set(training).difference({
        "device", "selection_metric", "log_epoch_history",
        "epoch_history_metrics",
    })
    if unknown_training:
        raise ValueError(f"Unknown training setting: {sorted(unknown_training)[0]}")
    if training.get("selection_metric") not in {"roc_auc", "average_precision"}:
        raise ValueError("Unknown selection metric")
    if not isinstance(training.get("device"), str) or not training["device"]:
        raise ValueError("training.device must be a non-empty string")
    if training.get("log_epoch_history") is not True:
        raise ValueError("training-policy ablation must log epoch history")
    if training.get("epoch_history_metrics") != [
        "training_loss", "validation_roc_auc"
    ]:
        raise ValueError("unsupported epoch_history_metrics")

    policies_raw = _require_unique_sequence(
        raw.get("training_policies"), "training_policies"
    )
    policies = tuple(
        _parse_training_policy(policy, index)
        for index, policy in enumerate(policies_raw)
    )
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("training policy names must be unique")

    defaults = _validate_defaults({
        "query_size": None,
        "query_fraction": splits["query_fraction"],
        "validation_query_size": None,
        "validation_query_fraction": splits["validation_query_fraction"],
        "train_ratio": splits["train_ratio"],
        "selection_metric": training["selection_metric"],
        "device": training["device"],
    })
    model_names = _require_unique_sequence(raw.get("models"), "models")
    if any(not isinstance(name, str) for name in model_names):
        raise ValueError("ablation models must be architecture names")
    models = tuple(
        _parse_model_grid({"type": name}, defaults, index)
        for index, name in enumerate(model_names)
    )
    return BenchmarkConfig(
        output=output,
        data_root=data_root,
        seeds=tuple(seeds),
        defaults=defaults,
        datasets=datasets,
        models=models,
        error_output=error_output,
        selected_configurations=_selected_configuration_path(
            raw.get("selected_configurations")
        ),
        kind="training-policy-ablation",
        training_policies=policies,
        epoch_history_output=epoch_output,
        log_epoch_history=True,
    )


def load_benchmark_config(path):
    """Load and validate a benchmark YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    raw = _require_mapping(raw, "benchmark configuration")
    raw_models = raw.get("models")
    if isinstance(raw_models, list):
        for index, model in enumerate(raw_models):
            if isinstance(model, str):
                raw_models[index] = resolve_model_reference(
                    model, base_dir=path.parent
                )
            elif isinstance(model, dict) and isinstance(model.get("type"), str):
                model["type"] = resolve_model_reference(
                    model["type"], base_dir=path.parent
                )
    if raw.get("kind") == "training-policy-ablation":
        return _load_training_policy_ablation(raw)
    unknown = set(raw).difference({
        "version", "output", "error_output", "data_root", "seeds",
        "defaults", "datasets", "models", "kind", "selected_configurations",
    })
    if unknown:
        raise ValueError(f"Unknown benchmark setting: {sorted(unknown)[0]}")
    version = raw.get("version")
    if version not in {1, 2}:
        raise ConfigurationError(
            "benchmark configuration version must be 1 or 2",
            component="benchmark-config",
        )

    selected_path = None
    selected_raw = raw.get("selected_configurations")
    if version == 1 and selected_raw is not None:
        raise ValueError("selected_configurations require configuration version 2")
    if version == 2:
        if raw.get("kind") != "selected-hyperparameter-benchmark":
            raise ValueError(
                "version-2 configurations must use kind "
                "'selected-hyperparameter-benchmark'"
            )
        selected_path = _selected_configuration_path(selected_raw)

    output = raw.get("output", "out/benchmark.csv")
    data_root = raw.get("data_root", "data")
    error_output = raw.get(
        "error_output", str(Path(output).with_suffix(".errors.jsonl"))
    )
    for value, name in (
        (output, "output"), (data_root, "data_root"),
        (error_output, "error_output"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    seeds = _require_unique_sequence(raw.get("seeds"), "seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be non-negative integers")
    defaults = _validate_defaults(raw.get("defaults", {}))
    datasets = tuple(
        _resolve_dataset_name(name)
        for name in _require_unique_sequence(raw.get("datasets"), "datasets")
    )
    raw_models = _require_unique_sequence(raw.get("models"), "models")
    if version == 2:
        if any(not isinstance(model, str) for model in raw_models):
            raise ValueError("version-2 models must be architecture names")
        raw_models = [{"type": model} for model in raw_models]
    models = tuple(
        _parse_model_grid(model, defaults, index)
        for index, model in enumerate(raw_models)
    )
    return BenchmarkConfig(
        output=output,
        data_root=data_root,
        seeds=tuple(seeds),
        defaults=defaults,
        datasets=datasets,
        models=models,
        error_output=error_output,
        selected_configurations=selected_path,
    )


_SELECTED_PARAMETER_KEYS = {
    "hidden_channels", "out_channels", "learning_rate", "weight_decay",
    "normalize_features", "conv_layer_type", "reduction_method",
    "regularization_type", "regularization_strength",
    "decoder_hidden_channels", "decoder_dropout", "kl_weight",
    "deal_profile", "theta_ss", "theta_aa", "theta_sa", "lambda_ss",
    "lambda_aa", "lambda_sa", "similarity_scale", "similarity_margin",
    "gatv2_kwargs",
}
_DEAL_PARAMETER_MAP = {
    "theta_ss": "deal_theta_ss",
    "theta_aa": "deal_theta_aa",
    "theta_sa": "deal_theta_sa",
    "lambda_ss": "deal_lambda_ss",
    "lambda_aa": "deal_lambda_aa",
    "lambda_sa": "deal_lambda_sa",
    "similarity_scale": "deal_similarity_scale",
    "similarity_margin": "deal_similarity_margin",
}


def _load_selected_configurations(path):
    """Load selections produced by the hyperparameter-search runner."""
    if not path.exists():
        raise FileNotFoundError(f"selected configurations not found: {path}")
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    document = _require_mapping(document, "selected configuration document")
    if document.get("version") != 1 or document.get("kind") != (
        "selected-hyperparameters"
    ):
        raise ValueError("expected a version-1 selected-hyperparameters document")
    return _require_mapping(document.get("selections"), "selections")


def _selected_experiment_job(config, dataset, model, selections):
    """Resolve one selected dataset–architecture pair into a job."""
    dataset_selections = _require_mapping(
        selections.get(dataset), f"selections.{dataset}"
    )
    selection = _require_mapping(
        dataset_selections.get(model.type),
        f"selections.{dataset}.{model.type}",
    )
    hyperparameters = _require_mapping(
        selection.get("hyperparameters"),
        f"selections.{dataset}.{model.type}.hyperparameters",
    )
    unknown = set(hyperparameters).difference(_SELECTED_PARAMETER_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown selected hyperparameter: {sorted(unknown)[0]}"
        )
    values = dict(hyperparameters)
    conv_layer = values.pop(
        "conv_layer_type", config.defaults["conv_layer_type"]
    )
    if conv_layer not in _CONV_LAYER_TYPES:
        raise ValueError("Unknown selected convolution layer")

    model_adapter = get_model_adapter(model.type)
    reducer = values.pop(
        "reduction_method", config.defaults["reduction_method"]
    )
    if model_adapter.capabilities.requires_reduced_features:
        if reducer not in available_reduction_preprocessors():
            raise ValueError("Unknown selected reduction method")
    elif "reduction_method" in hyperparameters:
        raise ValueError(
            f"{model.type} cannot use a selected reduction method"
        )

    regularization_type = values.pop("regularization_type", "none")
    regularization_strength = values.pop("regularization_strength", 0.0)
    regularization = _parse_regularization(
        [{"type": regularization_type, "strength": regularization_strength}],
        model.type,
    ) if model_adapter.capabilities.supports_agae_regularization else (
        RegularizationConfig("none", 0.0),
    )
    if not model_adapter.capabilities.supports_agae_regularization and (
        "regularization_type" in hyperparameters
        or "regularization_strength" in hyperparameters
    ):
        raise ValueError(f"{model.type} cannot use selected regularization")

    values.pop("deal_profile", None)
    for source, destination in _DEAL_PARAMETER_MAP.items():
        if source in values:
            if model.type != "deal":
                raise ValueError(f"{source} is only supported by DEAL")
            values[destination] = values.pop(source)

    parameters = {**config.defaults, **values}
    parameters.update({
        "out_filepath": config.output,
        "data_root": config.data_root,
    })
    _validate_job_parameters(parameters)
    if (
        not model_adapter.capabilities.trainable
        and parameters["lr_scheduler"] != "none"
    ):
        raise ValueError(f"{model.type} cannot use a learning-rate scheduler")
    return ExperimentJob(
        dataset=dataset,
        model_type=model.type,
        conv_layer_type=conv_layer,
        reduction_method=reducer,
        regularization=regularization[0],
        seeds=config.seeds,
        parameters=parameters,
    )


def expand_benchmark_config(config):
    """Expand a validated configuration into deterministic experiment jobs."""
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")
    if config.selected_configurations is not None:
        selections = _load_selected_configurations(
            config.selected_configurations
        )
        selected_jobs = tuple(
            _selected_experiment_job(config, dataset, model, selections)
            for dataset, model in product(config.datasets, config.models)
        )
        if not config.training_policies:
            return selected_jobs
        jobs = []
        for job, policy in product(selected_jobs, config.training_policies):
            parameters = {
                **job.parameters,
                "epochs": policy.max_epochs,
                "training_policy": policy.name,
                "checkpoint_strategy": policy.checkpoint_strategy,
                "early_stopping_patience": policy.early_stopping_patience,
                "early_stopping_min_delta": policy.early_stopping_min_delta,
                "early_stopping_minimum_epochs": (
                    policy.early_stopping_minimum_epochs
                ),
                "lr_scheduler": policy.lr_scheduler,
                "lr_scheduler_kwargs": policy.lr_scheduler_kwargs,
            }
            _validate_job_parameters(parameters)
            if not get_model_adapter(job.model_type).capabilities.trainable and (
                parameters["lr_scheduler"] != "none"
            ):
                raise ValueError(
                    f"{job.model_type} cannot use a learning-rate scheduler"
                )
            jobs.append(replace(job, parameters=parameters))
        return tuple(jobs)

    jobs = []
    for dataset, model in product(config.datasets, config.models):
        for conv_layer, reducer, regularizer in product(
            model.conv_layer_types,
            model.reduction_methods,
            model.regularization,
        ):
            parameters = {**config.defaults, **model.parameters}
            parameters.update({
                "out_filepath": config.output,
                "data_root": config.data_root,
            })
            _validate_job_parameters(parameters)
            if not get_model_adapter(model.type).capabilities.trainable and (
                parameters["lr_scheduler"] != "none"
            ):
                raise ValueError(
                    f"{model.type} cannot use a learning-rate scheduler"
                )
            jobs.append(ExperimentJob(
                dataset=dataset,
                model_type=model.type,
                conv_layer_type=conv_layer,
                reduction_method=reducer,
                regularization=regularizer,
                seeds=config.seeds,
                parameters=parameters,
            ))
    return tuple(jobs)


def load_completed_results(path):
    """Return completed ``(configuration_id, seed)`` pairs from a CSV file."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSVResultLogger.FIELDNAMES:
            raise ValueError("existing CSV schema does not match the logger schema")
        completed = set()
        for row in reader:
            if row["configuration_id"]:
                completed.add((row["configuration_id"], int(row["random_seed"])))
    return completed


def _training_configuration_id(job):
    """Return the ID of the trajectory shared by both test split rows."""
    parameters = {
        name: value for name, value in job.parameters.items()
        if name not in {"out_filepath", "data_root"}
    }
    return make_configuration_id({
        "dataset": job.dataset,
        "model_type": job.model_type,
        "conv_layer_type": job.conv_layer_type,
        "reduction_method": job.reduction_method,
        "regularization_type": job.regularization.type,
        "regularization_strength": job.regularization.strength,
        "parameters": parameters,
    })


def load_completed_epoch_history(path):
    """Return completed training-configuration, seed, and epoch tuples."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSVEpochHistoryLogger.FIELDNAMES:
            raise ValueError(
                "existing CSV schema does not match the epoch logger schema"
            )
        return {
            (
                row["configuration_id"],
                int(row["random_seed"]),
                int(row["epoch"]),
            )
            for row in reader
        }


def _append_failure(path, run_id, job, error):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "job": {
            "label": job.label,
            "dataset": job.dataset,
            "model_type": job.model_type,
            "conv_layer_type": job.conv_layer_type,
            "reduction_method": job.reduction_method,
            "regularization_type": job.regularization.type,
            "regularization_strength": job.regularization.strength,
            "seeds": list(job.seeds),
            "parameters": job.parameters,
        },
        "error_type": type(error).__name__,
        "message": str(error),
    }
    with path.open("a", encoding="utf-8") as error_file:
        error_file.write(json.dumps(record, sort_keys=True) + "\n")


def describe_benchmark(config, jobs):
    """Return a concise benchmark expansion summary."""
    return {
        "datasets": len(config.datasets),
        "jobs": len(jobs),
        "seeds_per_job": len(config.seeds),
        "model_runs": len(jobs) * len(config.seeds),
        "expected_result_rows": len(jobs) * len(config.seeds) * 2,
    }


def execute_benchmark(
    config, jobs, *, dry_run=False, resume=False, fail_fast=False,
    verbose=False, verbose_details=False, run_id=None, runner=run_experiment,
):
    """Execute expanded jobs sequentially and isolate job failures."""
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")
    jobs = tuple(jobs)
    benchmark_run_id = run_id or str(uuid4())
    expected_rows = len(jobs) * len(config.seeds) * 2
    if dry_run:
        return BenchmarkSummary(
            benchmark_run_id, len(jobs), 0, 0, expected_rows
        )

    completed = load_completed_results(config.output) if resume else set()
    epoch_logger = None
    completed_epochs = set()
    if config.log_epoch_history:
        epoch_logger = CSVEpochHistoryLogger(
            config.epoch_history_output, run_id=benchmark_run_id
        )
        if resume:
            completed_epochs = load_completed_epoch_history(
                config.epoch_history_output
            )
    succeeded = 0
    failed = 0
    show_progress = verbose or verbose_details
    compact_progress = verbose and not verbose_details
    progress = tqdm(
        total=(
            len(jobs) * len(config.seeds) if compact_progress else len(jobs)
        ),
        desc="Benchmark runs" if compact_progress else "Benchmark jobs",
        unit="run" if compact_progress else "job",
        disable=not show_progress,
        position=0,
    )
    progress_counts = {"ok": 0, "failed": 0, "skipped": 0}

    def update_display(label, stage=None):
        details = " ".join(
            f"{name}={value}" for name, value in progress_counts.items()
        )
        if stage is not None:
            details = f"{label} stage={stage} {details}"
        else:
            details = f"{label} {details}"
        progress.set_postfix_str(details)

    for job in jobs:
        accounted_runs = 0
        training_configuration_id = _training_configuration_id(job)

        def log_epoch(record):
            key = (
                training_configuration_id,
                record["random_seed"],
                record["epoch"],
            )
            if key in completed_epochs:
                return
            epoch_logger.log(EpochHistoryRecord(
                configuration_id=training_configuration_id,
                dataset=record["dataset"],
                model_type=record["model_type"],
                random_seed=record["random_seed"],
                training_policy=record["training_policy"],
                epoch=record["epoch"],
                training_loss=record["training_loss"],
                learning_rate=record["learning_rate"],
                validation_roc_auc=record["validation_roc_auc"],
                validation_average_precision=(
                    record["validation_average_precision"]
                ),
            ))
            completed_epochs.add(key)

        def handle_progress(event, **event_values):
            nonlocal accounted_runs
            dataset = event_values.get("dataset", job.dataset)
            model_type = event_values.get("model_type", job.model_type)
            seed = event_values.get("seed")
            label = f"{dataset}/{model_type}"
            if seed is not None:
                label += f"/seed={seed}"
            if event == "run_completed":
                progress_counts["ok"] += 1
                accounted_runs += 1
                progress.update(1)
                update_display(label)
            elif event == "run_skipped":
                progress_counts["skipped"] += 1
                accounted_runs += 1
                progress.update(1)
                update_display(label)
            else:
                update_display(label, event_values.get("stage"))

        if not compact_progress:
            progress.set_postfix_str(job.label)
        try:
            runner_kwargs = {
                "run_id": benchmark_run_id,
                "completed_results": completed,
            }
            if compact_progress:
                runner_kwargs.update({
                    "progress_callback": handle_progress,
                    "compact_progress": True,
                })
            namespace = job.to_namespace(verbose=show_progress)
            if epoch_logger is not None:
                namespace.epoch_callback = log_epoch
            runner(namespace, **runner_kwargs)
            succeeded += 1
            if compact_progress and accounted_runs < len(job.seeds):
                unreported = len(job.seeds) - accounted_runs
                progress_counts["ok"] += unreported
                progress.update(unreported)
                update_display(job.label)
        except Exception as error:
            failed += 1
            _append_failure(config.error_output, benchmark_run_id, job, error)
            if compact_progress:
                unreported = len(job.seeds) - accounted_runs
                progress_counts["failed"] += unreported
                progress.update(unreported)
                update_display(job.label)
                tqdm.write(
                    f"FAILED {job.label}: {type(error).__name__}: {error}"
                )
            if fail_fast:
                progress.close()
                raise
        finally:
            if not compact_progress:
                progress.update(1)
    progress.close()
    return BenchmarkSummary(
        benchmark_run_id, len(jobs), succeeded, failed, expected_rows
    )


def parse_args():
    """Parse benchmark-runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a benchmark YAML file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--verbose-details",
        action="store_true",
        help="show detailed nested experiment progress",
    )
    return parser.parse_args()


def main(cli_args=None):
    """Load, expand, summarize, and optionally execute a benchmark."""
    args = parse_args() if cli_args is None else cli_args
    config = load_benchmark_config(args.config)
    jobs = expand_benchmark_config(config)
    description = describe_benchmark(config, jobs)
    print(json.dumps(description, indent=2, sort_keys=True))
    if args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"{index:4d}. {job.label}")
    summary = execute_benchmark(
        config,
        jobs,
        dry_run=args.dry_run,
        resume=args.resume,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
        verbose_details=getattr(args, "verbose_details", False),
    )
    print("Benchmark complete")
    print(f"  Run ID: {summary.run_id}")
    print(
        f"  Jobs: {summary.jobs_succeeded} succeeded, "
        f"{summary.jobs_failed} failed, {summary.jobs_total} total"
    )
    print(f"  Expected result rows: {summary.expected_result_rows}")
    print(f"  Results: {config.output}")
    if summary.jobs_failed:
        print(f"  Errors: {config.error_output}")
    return summary


if __name__ == "__main__":
    main()
