"""Resolve canonical configurations into runtime experiment contracts."""

from __future__ import annotations

from itertools import product

from glpbench.configuration.ablation import AblationConfig
from glpbench.configuration.audit import AuditConfig
from glpbench.configuration.benchmark import BenchmarkConfig
from glpbench.configuration.validation import validate_config
from glpbench.contracts import ResolvedExperimentSpec, TrainingPolicy
from glpbench.models import get_model_registration


def _training_policy(training) -> TrainingPolicy:
    stopping = training.early_stopping
    return TrainingPolicy(
        max_epochs=training.max_epochs,
        optimizer="adam",
        optimizer_parameters={
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
        },
        checkpoint_strategy=training.checkpoint_strategy,
        selection_metric=training.selection_metric,
        early_stopping_patience=(stopping.patience if stopping.enabled else None),
        early_stopping_min_delta=stopping.min_delta,
        early_stopping_minimum_epochs=stopping.minimum_epochs,
        scheduler=training.scheduler.type,
        scheduler_parameters=training.scheduler.parameters,
    )


def _model_dimensions(model):
    capabilities = get_model_registration(model.architecture).capabilities
    conv_layers = (
        model.conv_layer_types or ("GCNConv",)
        if capabilities.uses_conv_layer else (None,)
    )
    reducers = (
        model.reduction_methods or ("pca",)
        if capabilities.requires_reduced_features else (None,)
    )
    return conv_layers, reducers


def _resolve(config, policies):
    specifications = []
    for dataset, model, seed, (policy_name, training) in product(
        config.datasets.names, config.models, config.execution.seeds, policies
    ):
        conv_layers, reducers = _model_dimensions(model)
        for conv_layer, reducer in product(conv_layers, reducers):
            model_parameters = dict(model.parameters)
            if conv_layer is not None:
                model_parameters["conv_layer_type"] = conv_layer
            if reducer is not None:
                model_parameters["reduction_method"] = reducer
            model_parameters.update({
                "learning_rate": training.learning_rate,
                "weight_decay": training.weight_decay,
                "training_policy": policy_name,
            })
            specifications.append(ResolvedExperimentSpec(
                schema_version=config.schema_version,
                dataset=dataset,
                model_type=model.architecture,
                protocol=config.split.protocol,
                seed=seed,
                device=config.execution.device,
                training_policy=_training_policy(training),
                dataset_parameters={
                    "root": str(config.datasets.root),
                    "normalize_features": config.datasets.normalize_features,
                },
                protocol_parameters={
                    "train_ratio": config.split.train_ratio,
                    "query_size": config.split.query_size,
                    "query_fraction": config.split.query_fraction,
                    "validation_query_size": config.split.validation_query_size,
                    "validation_query_fraction": (
                        config.split.validation_query_fraction
                    ),
                },
                preprocessing=reducer,
                preprocessing_parameters={},
                model_parameters=model_parameters,
            ))
    return tuple(specifications)


def resolve_experiment_specs(
    config: BenchmarkConfig | AblationConfig | AuditConfig,
) -> tuple[ResolvedExperimentSpec, ...]:
    """Expand a canonical runnable configuration into experiment specs."""
    validate_config(config)
    if isinstance(config, AblationConfig):
        policies = tuple(
            (policy.name, policy.training) for policy in config.policies
        )
    elif isinstance(config, AuditConfig):
        policies = (("convergence-audit", config.training),)
    elif isinstance(config, BenchmarkConfig):
        policies = (("default", config.training),)
    else:
        raise TypeError("config must be a benchmark, ablation, or audit config")
    return _resolve(config, policies)


__all__ = ["resolve_experiment_specs"]
