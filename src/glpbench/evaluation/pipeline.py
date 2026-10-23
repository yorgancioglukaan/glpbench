"""Shared data preparation for benchmark and search experiment paths."""

import torch
from torch_geometric.data import Data

from glpbench.models import get_model_adapter
from glpbench.preprocessing import (
    PreprocessingPipeline,
    PreprocessingStep,
)
from glpbench.contracts import PreparationResult, SplitBundle


def _prepared_bundle(source, partitions):
    """Rebuild a split bundle around cloned, preprocessed partitions."""
    return SplitBundle(
        train=partitions[0],
        validation=partitions[1],
        tests=partitions[2],
        metadata=source.metadata,
        validation_query_nodes=source.validation_query_nodes,
        test_query_nodes=source.test_query_nodes,
        visible_node_mask=source.visible_node_mask,
    )


def _prepare_features(
    source_data,
    fit_mask,
    *,
    requires_reduced_features,
    normalize_features,
    out_channels,
    reduction_method,
    random_state,
):
    """Resolve the compatibility options into an ordered preprocessing run."""
    steps = [PreprocessingStep(
        "normalize-features" if normalize_features else "identity"
    )]
    if requires_reduced_features:
        steps.append(PreprocessingStep(
            reduction_method, {"n_components": out_channels}
        ))
    return PreprocessingPipeline(steps).fit_transform(
        source_data.x,
        fit_mask,
        seed=random_state,
    )


def prepare_candidate_data(
    source_data,
    splits,
    *,
    model_type,
    normalize_features,
    out_channels,
    reduction_method="pca",
    random_state=None,
):
    """Prepare isolated candidate data from one immutable split foundation."""
    if not isinstance(source_data, Data):
        raise TypeError("source_data must be a torch_geometric Data object")
    if not isinstance(splits, SplitBundle):
        raise TypeError("splits must be a SplitBundle")
    if splits.validation is None or set(splits.tests) != {
        "transductive", "semi-inductive"
    }:
        raise ValueError("candidate preparation requires paired test splits")
    if not isinstance(normalize_features, bool):
        raise TypeError("normalize_features must be boolean")
    model_adapter = get_model_adapter(model_type)

    feature_data = _prepare_features(
        source_data,
        splits.train.visible_node_mask,
        requires_reduced_features=(
            model_adapter.capabilities.requires_reduced_features
        ),
        normalize_features=normalize_features,
        out_channels=out_channels,
        reduction_method=reduction_method,
        random_state=random_state,
    )

    prepared_splits = [
        splits.train.clone(),
        splits.validation.clone(),
        splits.tests["transductive"].clone(),
        splits.tests["semi-inductive"].clone(),
    ]
    for split in prepared_splits:
        split.x = feature_data.features.clone()

    if feature_data.reduced_features is not None:
        for split in prepared_splits:
            split.x_reduced = feature_data.reduced_features.clone()

    prepared_bundle = _prepared_bundle(
        splits,
        (
            prepared_splits[0],
            prepared_splits[1],
            {
                "transductive": prepared_splits[2],
                "semi-inductive": prepared_splits[3],
            },
        ),
    )
    return PreparationResult(
        splits=prepared_bundle,
        preprocessing_time_seconds=feature_data.fit_time_seconds,
    )


def prepare_validation_data(
    source_data,
    splits,
    *,
    model_type,
    normalize_features,
    out_channels,
    reduction_method="pca",
    random_state=None,
):
    """Prepare training and validation data without exposing test splits."""
    if not isinstance(splits, SplitBundle):
        raise TypeError("splits must be a SplitBundle")
    if splits.tests:
        prepared = prepare_candidate_data(
            source_data,
            splits,
            model_type=model_type,
            normalize_features=normalize_features,
            out_channels=out_channels,
            reduction_method=reduction_method,
            random_state=random_state,
        )
        validation_bundle = _prepared_bundle(
            prepared.splits,
            (
                prepared.splits.train,
                prepared.splits.validation,
                {},
            ),
        )
        return PreparationResult(
            validation_bundle,
            prepared.preprocessing_time_seconds,
        )
    if not isinstance(source_data, Data):
        raise TypeError("source_data must be a torch_geometric Data object")
    if not isinstance(normalize_features, bool):
        raise TypeError("normalize_features must be boolean")

    model_adapter = get_model_adapter(model_type)
    if splits.validation is None:
        raise ValueError("validation preparation requires a validation split")
    train = splits.train.clone()
    validation = splits.validation.clone()
    feature_data = _prepare_features(
        source_data,
        train.visible_node_mask,
        requires_reduced_features=(
            model_adapter.capabilities.requires_reduced_features
        ),
        normalize_features=normalize_features,
        out_channels=out_channels,
        reduction_method=reduction_method,
        random_state=random_state,
    )
    train.x = feature_data.features.clone()
    validation.x = feature_data.features.clone()
    if feature_data.reduced_features is not None:
        train.x_reduced = feature_data.reduced_features.clone()
        validation.x_reduced = feature_data.reduced_features.clone()
    return PreparationResult(
        _prepared_bundle(splits, (train, validation, {})),
        feature_data.fit_time_seconds,
    )
