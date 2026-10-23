"""Feature preprocessing and dimensionality reduction."""

from glpbench.preprocessing.normalization import normalize_graph_features
from glpbench.preprocessing.pipeline import (
    FeaturePreparation,
    PreprocessingPipeline,
    PreprocessingStep,
)
from glpbench.preprocessing.registry import (
    available_preprocessors,
    available_reduction_preprocessors,
    create_preprocessor,
    register_preprocessor,
)
from glpbench.preprocessing.reduction import (
    attach_reduced_features,
    available_reduction_methods,
    fit_reducer,
    register_reduction_method,
    transform_features,
)

__all__ = [
    "attach_reduced_features",
    "available_preprocessors",
    "available_reduction_preprocessors",
    "available_reduction_methods",
    "create_preprocessor",
    "FeaturePreparation",
    "fit_reducer",
    "normalize_graph_features",
    "PreprocessingPipeline",
    "PreprocessingStep",
    "register_preprocessor",
    "register_reduction_method",
    "transform_features",
]
