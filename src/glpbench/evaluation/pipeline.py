"""Shared data preparation for benchmark and search experiment paths."""

from dataclasses import dataclass
import time

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from glpbench.models import get_model_spec
from glpbench.protocols import (
    ComparableSplits,
    ValidationSplits,
    calculate_reduced_features,
)


@dataclass(frozen=True)
class ValidationData:
    """Expose only the training and validation data used during selection."""

    train: Data
    validation: Data
    reducer_fit_time_seconds: float | None


@dataclass(frozen=True)
class CandidateData:
    """Store fully preprocessed data for validation and final evaluation."""

    train: Data
    validation: Data
    test_transductive: Data
    test_inductive: Data
    reducer_fit_time_seconds: float | None

    def for_validation(self):
        """Return a restricted view that contains no test splits."""
        return ValidationData(
            self.train,
            self.validation,
            self.reducer_fit_time_seconds,
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
    if not isinstance(splits, ComparableSplits):
        raise TypeError("splits must be a ComparableSplits object")
    if not isinstance(normalize_features, bool):
        raise TypeError("normalize_features must be boolean")
    model_spec = get_model_spec(model_type)

    feature_data = source_data.clone()
    if normalize_features:
        feature_data = T.NormalizeFeatures()(feature_data)

    prepared_splits = [split.clone() for split in splits]
    for split in prepared_splits:
        split.x = feature_data.x.clone()

    reducer_fit_time_seconds = None
    if model_spec.uses_reduced_features:
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

    return CandidateData(
        train=prepared_splits[0],
        validation=prepared_splits[1],
        test_transductive=prepared_splits[2],
        test_inductive=prepared_splits[3],
        reducer_fit_time_seconds=reducer_fit_time_seconds,
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
    if not isinstance(splits, ValidationSplits):
        return prepare_candidate_data(
            source_data,
            splits,
            model_type=model_type,
            normalize_features=normalize_features,
            out_channels=out_channels,
            reduction_method=reduction_method,
            random_state=random_state,
        ).for_validation()
    if not isinstance(source_data, Data):
        raise TypeError("source_data must be a torch_geometric Data object")
    if not isinstance(normalize_features, bool):
        raise TypeError("normalize_features must be boolean")

    model_spec = get_model_spec(model_type)
    feature_data = source_data.clone()
    if normalize_features:
        feature_data = T.NormalizeFeatures()(feature_data)
    train, validation = (split.clone() for split in splits)
    train.x = feature_data.x.clone()
    validation.x = feature_data.x.clone()

    reducer_fit_time_seconds = None
    if model_spec.uses_reduced_features:
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
    return ValidationData(train, validation, reducer_fit_time_seconds)
