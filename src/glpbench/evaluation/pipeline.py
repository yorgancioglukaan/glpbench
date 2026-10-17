"""Shared data preparation for benchmark and search experiment paths."""

import time

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from glpbench.models import get_model_registration
from glpbench.protocols import (
    calculate_reduced_features,
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
    model_registration = get_model_registration(model_type)

    feature_data = source_data.clone()
    if normalize_features:
        feature_data = T.NormalizeFeatures()(feature_data)

    prepared_splits = [
        splits.train.clone(),
        splits.validation.clone(),
        splits.tests["transductive"].clone(),
        splits.tests["semi-inductive"].clone(),
    ]
    for split in prepared_splits:
        split.x = feature_data.x.clone()

    reducer_fit_time_seconds = None
    if model_registration.capabilities.requires_reduced_features:
        fit_nodes = prepared_splits[0].visible_node_mask.detach().cpu()
        reducer_start = time.perf_counter()
        calculate_reduced_features(
            feature_data,
            dim=out_channels,
            reduction_method=reduction_method,
            fit_nodes=fit_nodes,
            random_state=random_state,
        )
        reducer_fit_time_seconds = time.perf_counter() - reducer_start
        for split in prepared_splits:
            split.x_reduced = feature_data.x_reduced.clone()

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
        preprocessing_time_seconds=reducer_fit_time_seconds,
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

    model_registration = get_model_registration(model_type)
    feature_data = source_data.clone()
    if normalize_features:
        feature_data = T.NormalizeFeatures()(feature_data)
    if splits.validation is None:
        raise ValueError("validation preparation requires a validation split")
    train = splits.train.clone()
    validation = splits.validation.clone()
    train.x = feature_data.x.clone()
    validation.x = feature_data.x.clone()

    reducer_fit_time_seconds = None
    if model_registration.capabilities.requires_reduced_features:
        reducer_start = time.perf_counter()
        calculate_reduced_features(
            feature_data,
            dim=out_channels,
            reduction_method=reduction_method,
            fit_nodes=train.visible_node_mask.detach().cpu(),
            random_state=random_state,
        )
        reducer_fit_time_seconds = time.perf_counter() - reducer_start
        train.x_reduced = feature_data.x_reduced.clone()
        validation.x_reduced = feature_data.x_reduced.clone()
    return PreparationResult(
        _prepared_bundle(splits, (train, validation, {})),
        reducer_fit_time_seconds,
    )
