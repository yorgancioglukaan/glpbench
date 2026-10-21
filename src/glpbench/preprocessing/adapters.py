"""Built-in adapters implementing the feature-preprocessor contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from glpbench.preprocessing.normalization import normalize_graph_features
from glpbench.preprocessing.reduction import fit_reducer, transform_features


@dataclass
class IdentityPreprocessor:
    """Return an isolated copy of the input features."""

    name: str = "identity"
    requires_fitting: bool = False
    supports_sparse_inputs: bool = True
    uses_randomness: bool = False
    output: str = "features"

    def fit(self, x, fit_mask, seed=None):
        """Validate the feature matrix without learning state."""
        self._validate(x)
        return self

    def transform(self, x):
        """Return a clone of ``x``."""
        self._validate(x)
        return x.clone()

    @staticmethod
    def _validate(x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("features must be a two-dimensional tensor")


@dataclass
class NormalizeFeaturesPreprocessor:
    """Apply PyG row-wise feature normalization."""

    name: str = "normalize-features"
    requires_fitting: bool = False
    supports_sparse_inputs: bool = False
    uses_randomness: bool = False
    output: str = "features"

    def fit(self, x, fit_mask, seed=None):
        """Validate the feature matrix without learning state."""
        IdentityPreprocessor._validate(x)
        return self

    def transform(self, x):
        """Return normalized features without modifying ``x``."""
        graph = Data(x=x, num_nodes=x.size(0))
        return normalize_graph_features(graph).x


@dataclass
class ReductionPreprocessor:
    """Adapt a registered dimensionality reducer to the common interface."""

    method: str
    n_components: int
    parameters: dict = field(default_factory=dict)
    name: str = field(init=False)
    requires_fitting: bool = field(default=True, init=False)
    supports_sparse_inputs: bool = field(default=True, init=False)
    uses_randomness: bool = field(init=False)
    output: str = field(default="reduced_features", init=False)
    reducer: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.name = self.method
        self.uses_randomness = self.method in {
            "pca", "sparse-random-projection", "truncated-svd"
        }

    def fit(self, x, fit_mask, seed=None):
        """Fit the wrapped reducer using only rows selected by ``fit_mask``."""
        self.reducer = fit_reducer(
            x,
            fit_nodes=fit_mask,
            method=self.method,
            n_components=self.n_components,
            random_state=seed,
            **self.parameters,
        )
        return self

    def transform(self, x):
        """Transform all feature rows using the fitted reducer."""
        if self.reducer is None:
            raise RuntimeError("preprocessor must be fitted before transform")
        return transform_features(x, self.reducer)


__all__ = [
    "IdentityPreprocessor",
    "NormalizeFeaturesPreprocessor",
    "ReductionPreprocessor",
]
