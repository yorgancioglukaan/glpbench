"""Dataset providers, metadata, and graph validation."""

from glpbench.interfaces import DatasetProvider
from glpbench.datasets.builtin import load_dataset
from glpbench.datasets.registry import (
    available_datasets,
    get_dataset_provider,
    register_dataset_provider,
)
from glpbench.datasets.validation import validate_graph_data

__all__ = [
    "DatasetProvider",
    "available_datasets",
    "get_dataset_provider",
    "load_dataset",
    "register_dataset_provider",
    "validate_graph_data",
]
