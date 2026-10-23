"""Semantic validation for canonical configuration contracts."""

from __future__ import annotations

from typing import Any

from glpbench.configuration.ablation import AblationConfig
from glpbench.configuration.audit import AuditConfig
from glpbench.configuration.benchmark import BenchmarkConfig
from glpbench.configuration.parsing import configuration_error
from glpbench.configuration.search import SearchConfig
from glpbench.datasets import available_datasets
from glpbench.models import available_model_types, get_model_adapter
from glpbench.preprocessing import available_reduction_preprocessors
from glpbench.training import validate_scheduler_config


CanonicalConfig = BenchmarkConfig | SearchConfig | AblationConfig | AuditConfig
_CONV_LAYERS = {"GCNConv", "GraphConv", "SAGEConv", "GATv2Conv"}
_SPLIT_PROTOCOLS = {
    "matched-semi-inductive",
    "matched-semi-inductive-validation-only",
}
_DEVICES = {"cpu", "cuda", "mps"}
_MODEL_PARAMETERS = {
    "hidden_channels", "out_channels", "learning_rate", "weight_decay",
    "decoder_hidden_channels", "decoder_dropout", "kl_weight",
    "regularization_type", "regularization_strength", "gatv2_kwargs",
    "deal_theta_ss", "deal_theta_aa", "deal_theta_sa", "deal_lambda_ss",
    "deal_lambda_aa", "deal_lambda_sa", "deal_similarity_scale",
    "deal_similarity_margin",
}


def _validate_dataset_names(config: CanonicalConfig) -> None:
    supported = set(available_datasets())
    for index, name in enumerate(config.datasets.names):
        if name not in supported:
            raise configuration_error(
                ("datasets", "names", index), f"references unknown dataset {name!r}"
            )


def _validate_execution(config: CanonicalConfig) -> None:
    if config.execution.device not in _DEVICES:
        raise configuration_error(
            ("execution", "device"), "must be cpu, cuda, or mps"
        )


def _validate_split(config: CanonicalConfig) -> None:
    if config.split.protocol not in _SPLIT_PROTOCOLS:
        raise configuration_error(
            ("split", "protocol"),
            f"references unsupported protocol {config.split.protocol!r}",
        )


def _validate_training(training, path: tuple[str | int, ...]) -> None:
    if training.checkpoint_strategy not in {"best-validation", "final"}:
        raise configuration_error(
            path + ("checkpoint_strategy",),
            "must be best-validation or final",
        )
    if training.selection_metric not in {"roc_auc", "average_precision"}:
        raise configuration_error(
            path + ("selection_metric",), "must be roc_auc or average_precision"
        )
    try:
        validate_scheduler_config(
            training.scheduler.type,
            dict(training.scheduler.parameters),
            epochs=training.max_epochs,
        )
    except (TypeError, ValueError) as error:
        raise configuration_error(path + ("scheduler",), str(error)) from error


def _validate_model(
    model, path: tuple[str | int, ...], *, search: bool = False
) -> None:
    if model.architecture not in set(available_model_types()):
        raise configuration_error(
            path + ("architecture",),
            f"references unknown model {model.architecture!r}",
        )
    capabilities = get_model_adapter(model.architecture).capabilities
    conv_layers = () if search else model.conv_layer_types
    reduction_methods = () if search else model.reduction_methods
    for index, name in enumerate(conv_layers):
        if name not in _CONV_LAYERS:
            raise configuration_error(
                path + ("conv_layer_types", index),
                f"references unknown convolution {name!r}",
            )
    if conv_layers and not capabilities.uses_conv_layer:
        raise configuration_error(
            path + ("conv_layer_types",),
            f"is not supported by {model.architecture!r}",
        )
    reducers = set(available_reduction_preprocessors())
    for index, name in enumerate(reduction_methods):
        if name not in reducers:
            raise configuration_error(
                path + ("reduction_methods", index),
                f"references unknown reducer {name!r}",
            )
    if reduction_methods and not capabilities.requires_reduced_features:
        raise configuration_error(
            path + ("reduction_methods",),
            f"is not supported by {model.architecture!r}",
        )
    parameters = model.spaces if search else model.parameters
    for name in parameters:
        if name not in _MODEL_PARAMETERS and name not in {
            "conv_layer_type", "reduction_method"
        }:
            key = "spaces" if search else "parameters"
            raise configuration_error(
                path + (key, name), "is not a recognized model parameter"
            )
    if search and "conv_layer_type" in parameters:
        for index, name in enumerate(parameters["conv_layer_type"].values):
            if name not in _CONV_LAYERS:
                raise configuration_error(
                    path + ("spaces", "conv_layer_type", "values", index),
                    f"references unknown convolution {name!r}",
                )
        if not capabilities.uses_conv_layer:
            raise configuration_error(
                path + ("spaces", "conv_layer_type"),
                f"is not supported by {model.architecture!r}",
            )
    if search and "reduction_method" in parameters:
        reducers = set(available_reduction_preprocessors())
        for index, name in enumerate(parameters["reduction_method"].values):
            if name not in reducers:
                raise configuration_error(
                    path + ("spaces", "reduction_method", "values", index),
                    f"references unknown reducer {name!r}",
                )
        if not capabilities.requires_reduced_features:
            raise configuration_error(
                path + ("spaces", "reduction_method"),
                f"is not supported by {model.architecture!r}",
            )


def validate_config(config: CanonicalConfig) -> CanonicalConfig:
    """Validate registered names and cross-component compatibility."""
    if not isinstance(config, (BenchmarkConfig, SearchConfig, AblationConfig, AuditConfig)):
        raise TypeError("config must be a canonical GLPBench configuration")
    _validate_dataset_names(config)
    _validate_execution(config)
    _validate_split(config)
    if isinstance(config, AblationConfig):
        for index, policy in enumerate(config.policies):
            _validate_training(policy.training, ("policies", index, "training"))
    else:
        _validate_training(config.training, ("training",))
    for index, model in enumerate(config.models):
        _validate_model(
            model, ("models", index), search=isinstance(config, SearchConfig)
        )
        if isinstance(config, (SearchConfig, AuditConfig)) and not (
            get_model_adapter(model.architecture).capabilities.trainable
        ):
            raise configuration_error(
                ("models", index, "architecture"),
                "must be trainable for this configuration kind",
            )
    if isinstance(config, SearchConfig):
        if config.selection.aggregation != "mean":
            raise configuration_error(("selection", "aggregation"), "must be 'mean'")
        if config.selection.validation_setting != "semi-inductive":
            raise configuration_error(
                ("selection", "validation_setting"), "must be 'semi-inductive'"
            )
    return config


__all__ = ["CanonicalConfig", "validate_config"]
