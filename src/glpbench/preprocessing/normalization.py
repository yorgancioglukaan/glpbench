"""Stateless node-feature normalization."""

from __future__ import annotations

from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures


def normalize_graph_features(data):
    """Return a graph clone with row-normalized node features.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Attributed graph whose features should be normalized.

    Returns
    -------
    torch_geometric.data.Data
        A normalized clone. The input graph is not modified.
    """
    if not isinstance(data, Data):
        raise TypeError("data must be a torch_geometric.data.Data object")
    if data.x is None:
        raise ValueError("data must contain node features")
    return NormalizeFeatures()(data.clone())


__all__ = ["normalize_graph_features"]
