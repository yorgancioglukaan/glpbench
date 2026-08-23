"""Fit and apply node-feature dimensionality reducers."""

from collections.abc import Callable

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection


_REDUCER_FACTORIES: dict[str, Callable] = {
    "pca": PCA,
    "sparse-random-projection": SparseRandomProjection,
    "truncated-svd": TruncatedSVD,
}


def register_reduction_method(name, factory):
    """Register a reducer constructor under a stable name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("reduction method name must be a non-empty string")
    if not callable(factory):
        raise TypeError("reducer factory must be callable")
    normalized_name = name.strip().lower()
    if normalized_name in _REDUCER_FACTORIES:
        raise ValueError(f"reduction method '{normalized_name}' is already registered")
    _REDUCER_FACTORIES[normalized_name] = factory


def available_reduction_methods():
    """Return registered reduction method names in stable order."""
    return tuple(sorted(_REDUCER_FACTORIES))


def _as_numpy(features):
    """Convert a feature tensor to a CPU NumPy array."""
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.ndim != 2:
        raise ValueError("features must have shape [num_nodes, num_features]")
    if features.layout != torch.strided:
        features = features.to_dense()
    return features.detach().cpu().numpy()


def _training_features(features, fit_nodes):
    """Select the rows used to fit a reducer."""
    if fit_nodes is None:
        return features
    node_index = torch.as_tensor(fit_nodes, device=features.device)
    if node_index.dtype == torch.bool:
        if node_index.ndim != 1 or node_index.numel() != features.size(0):
            raise ValueError("boolean fit_nodes must have one entry per node")
    else:
        node_index = node_index.to(torch.long)
        if node_index.ndim != 1:
            raise ValueError("fit_nodes must be a one-dimensional mask or index")
    selected = features[node_index]
    if selected.size(0) == 0:
        raise ValueError("fit_nodes must select at least one node")
    return selected


def fit_reducer(
    features,
    fit_nodes=None,
    method="pca",
    n_components=128,
    random_state=None,
    **kwargs,
):
    """Fit a registered reducer on selected node features."""
    if method not in _REDUCER_FACTORIES:
        choices = ", ".join(available_reduction_methods())
        raise ValueError(f"Unknown reduction method '{method}'. Available: {choices}.")
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")

    training_features = _training_features(features, fit_nodes)
    factory = _REDUCER_FACTORIES[method]
    reducer = factory(
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )
    reducer.fit(_as_numpy(training_features))
    reducer.reduction_method = method
    return reducer


def transform_features(features, reducer):
    """Transform node features without refitting the reducer."""
    if not hasattr(reducer, "transform") or not callable(reducer.transform):
        raise TypeError("reducer must provide a callable transform method")
    transformed = reducer.transform(_as_numpy(features))
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    if transformed.ndim != 2 or transformed.shape[0] != features.size(0):
        raise ValueError("reducer output must have one row per node")
    return torch.as_tensor(
        transformed, dtype=torch.float, device=features.device
    )


def attach_reduced_features(data, reducer):
    """Attach a reducer and its transformed node features to a graph."""
    if data.x is None:
        raise ValueError("data must contain node features")
    data.x_reduced = transform_features(data.x, reducer)
    data.feature_reducer = reducer
    data.dim_reduction = reducer
    data.reduction_method = getattr(reducer, "reduction_method", None)
    return data
