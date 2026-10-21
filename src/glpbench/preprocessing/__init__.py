"""Feature preprocessing and dimensionality reduction."""

from glpbench.preprocessing.normalization import normalize_graph_features
from glpbench.preprocessing.reduction import (
    attach_reduced_features,
    available_reduction_methods,
    fit_reducer,
    register_reduction_method,
    transform_features,
)

__all__ = [
    "attach_reduced_features",
    "available_reduction_methods",
    "fit_reducer",
    "normalize_graph_features",
    "register_reduction_method",
    "transform_features",
]
