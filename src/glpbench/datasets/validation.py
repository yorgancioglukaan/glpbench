"""Validation for graph datasets accepted by GLPBench."""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

from glpbench.exceptions import DatasetError


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _invalid(message, dataset):
    """Raise a contextualized dataset-validation error."""
    raise DatasetError(
        message,
        component="dataset-validator",
        dataset=dataset,
    )


def validate_graph_data(
    data,
    *,
    dataset=None,
    require_features=True,
    require_undirected=True,
    require_simple=True,
):
    """Validate a homogeneous attributed graph without modifying it.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph to validate.
    dataset : str, optional
        Dataset name included in validation errors.
    require_features : bool, optional
        Whether a two-dimensional numeric feature matrix is required.
    require_undirected : bool, optional
        Whether every edge must have its reverse orientation.
    require_simple : bool, optional
        Whether self-loops and duplicate directed edges are forbidden.

    Returns
    -------
    torch_geometric.data.Data
        The unchanged input graph.
    """
    if not isinstance(data, Data):
        _invalid("graph must be a torch_geometric.data.Data object", dataset)
    for value, name in (
        (require_features, "require_features"),
        (require_undirected, "require_undirected"),
        (require_simple, "require_simple"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    if dataset is not None and (
        not isinstance(dataset, str) or not dataset.strip()
    ):
        raise ValueError("dataset must be a non-empty string when provided")

    num_nodes = data.num_nodes
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
        _invalid("graph must define an unambiguous integer node count", dataset)
    if num_nodes <= 0:
        _invalid("graph must contain at least one node", dataset)

    features = getattr(data, "x", None)
    if require_features and features is None:
        _invalid("graph must contain node features in data.x", dataset)
    if features is not None:
        if not isinstance(features, torch.Tensor):
            _invalid("data.x must be a torch.Tensor", dataset)
        if features.ndim != 2:
            _invalid("data.x must have shape [num_nodes, num_features]", dataset)
        if features.size(0) != num_nodes:
            _invalid("data.x must contain one row per node", dataset)
        if features.size(1) <= 0:
            _invalid("data.x must contain at least one feature", dataset)
        if features.dtype == torch.bool or features.is_complex() or (
            not features.is_floating_point()
            and features.dtype not in _INTEGER_DTYPES
        ):
            _invalid("data.x must have a real numeric dtype", dataset)
        if not bool(torch.isfinite(features).all()):
            _invalid("data.x must contain only finite values", dataset)

    edge_index = getattr(data, "edge_index", None)
    if not isinstance(edge_index, torch.Tensor):
        _invalid("graph must contain edge_index as a torch.Tensor", dataset)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        _invalid("edge_index must have shape [2, num_edges]", dataset)
    if edge_index.dtype not in _INTEGER_DTYPES:
        _invalid("edge_index must have an integer dtype", dataset)
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            _invalid("edge_index contains a node outside the graph", dataset)
    if require_simple:
        source, target = edge_index
        if bool((source == target).any()):
            _invalid("self-loops are not supported", dataset)
        if torch.unique(edge_index, dim=1).size(1) != edge_index.size(1):
            _invalid("duplicate directed edges are not supported", dataset)
    if require_undirected and not is_undirected(edge_index, num_nodes=num_nodes):
        _invalid("graph must be undirected with symmetric edges", dataset)
    return data


__all__ = ["validate_graph_data"]
