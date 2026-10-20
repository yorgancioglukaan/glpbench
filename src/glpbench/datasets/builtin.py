"""Built-in benchmark dataset providers and loading facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Amazon,
    CitationFull,
    Coauthor,
    DeezerEurope,
    Planetoid,
    WikiCS,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops

from glpbench.contracts import DatasetMetadata, LoadedDataset
from glpbench.exceptions import DatasetError
from glpbench.datasets.registry import (
    available_datasets,
    get_dataset_provider,
    normalize_dataset_name,
    register_dataset_provider,
)
from glpbench.datasets.validation import validate_graph_data


_PLANETOID_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def _load_planetoid(name, root):
    """Load one graph from the Planetoid family."""
    dataset = Planetoid(root=str(root), name=_PLANETOID_NAMES[name])
    return dataset[0].clone()


def _load_amazon(name, root):
    """Load an Amazon co-purchase graph."""
    dataset_name = "Photo" if name == "amazon-photo" else "Computers"
    dataset = Amazon(root=str(root), name=dataset_name)
    return dataset[0].clone()


def _load_coauthor(name, root):
    """Load the Coauthor CS graph."""
    dataset = Coauthor(root=str(root), name="CS")
    return dataset[0].clone()


def _load_citation_full(name, root):
    """Load the full DBLP citation graph as undirected."""
    dataset = CitationFull(root=str(root), name="DBLP", to_undirected=True)
    return dataset[0].clone()


def _load_wikics(name, root):
    """Load WikiCS as an undirected graph."""
    dataset = WikiCS(root=str(root), is_undirected=True)
    return dataset[0].clone()


def _load_deezer(name, root):
    """Load the Deezer Europe graph."""
    dataset = DeezerEurope(root=str(root))
    return dataset[0].clone()


@dataclass(frozen=True)
class _BuiltinDatasetProvider:
    """Adapt one existing PyG loader to the dataset-provider contract."""

    name: str
    canonical_name: str
    family: str
    loader: Callable[[str, Path], Data]

    def load(self, root, device):
        """Load, explicitly standardize, validate, and describe the graph."""
        data = self.loader(self.name, Path(root))
        if not isinstance(data, Data):
            raise DatasetError(
                "dataset loader must return torch_geometric.data.Data",
                component="dataset-provider",
                dataset=self.name,
            )
        preprocessing = ["remove-self-loops"]
        edge_index = getattr(data, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            raise DatasetError(
                "loaded dataset does not contain edge_index",
                component="dataset-provider",
                dataset=self.name,
            )
        edge_attr = getattr(data, "edge_attr", None)
        data.edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        if getattr(data, "edge_attr", None) is not None:
            data.edge_attr = edge_attr
        validate_graph_data(data, dataset=self.name)
        data = data.to(device)
        metadata = DatasetMetadata(
            name=self.canonical_name,
            family=self.family,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
            directed=False,
            has_node_features=True,
            num_node_features=data.num_features,
            loader_preprocessing=tuple(preprocessing),
        )
        return LoadedDataset(data=data, metadata=metadata)


def _register_builtin_providers():
    """Register the bundled PyG-backed providers once at import time."""
    specifications = [
        *(
            (name, canonical, "Planetoid", _load_planetoid)
            for name, canonical in _PLANETOID_NAMES.items()
        ),
        ("amazon-photo", "AmazonPhoto", "Amazon", _load_amazon),
        ("amazon-computers", "AmazonComputers", "Amazon", _load_amazon),
        ("coauthor-cs", "CoauthorCS", "Coauthor", _load_coauthor),
        ("dblp", "DBLP", "CitationFull", _load_citation_full),
        ("wikics", "WikiCS", "WikiCS", _load_wikics),
        ("deezer-europe", "DeezerEurope", "Deezer", _load_deezer),
    ]
    for name, canonical_name, family, loader in specifications:
        register_dataset_provider(
            _BuiltinDatasetProvider(name, canonical_name, family, loader),
            replace=True,
        )


_register_builtin_providers()


def _validated_provider_output(provider, loaded, target_device):
    """Validate and isolate one provider result before public use."""
    if not isinstance(loaded, LoadedDataset):
        raise DatasetError(
            "dataset provider must return LoadedDataset",
            component="dataset-provider",
            dataset=provider.name,
        )
    if normalize_dataset_name(loaded.metadata.name) != normalize_dataset_name(
        provider.name
    ):
        raise DatasetError(
            "provider name does not match returned dataset metadata",
            component="dataset-provider",
            dataset=provider.name,
        )
    data = loaded.data.clone().to(target_device)
    validate_graph_data(data, dataset=provider.name)
    return LoadedDataset(data=data, metadata=loaded.metadata)


def load_dataset(name, root="data", normalize_features=True, device="cpu"):
    """Load and standardize a registered benchmark graph.

    Parameters
    ----------
    name : str
        Registered dataset name or alias. Matching ignores case and separators.
    root : str or pathlib.Path, optional
        Directory used for downloaded and cached data.
    normalize_features : bool, optional
        Whether to normalize node features row-wise.
    device : str or torch.device, optional
        Device receiving the returned graph.

    Returns
    -------
    glpbench.contracts.LoadedDataset
        Validated PyG graph and dataset metadata.
    """
    if not isinstance(root, (str, Path)):
        raise TypeError("root must be a string or pathlib.Path")
    if not isinstance(normalize_features, bool):
        raise TypeError("normalize_features must be a bool")
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a string or torch.device")
    try:
        target_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid device '{device}'") from error

    provider = get_dataset_provider(name)
    loaded = provider.load(Path(root), target_device)
    loaded = _validated_provider_output(provider, loaded, target_device)
    if not normalize_features:
        return loaded

    data = NormalizeFeatures()(loaded.data.clone())
    validate_graph_data(data, dataset=provider.name)
    metadata = replace(
        loaded.metadata,
        loader_preprocessing=(
            *loaded.metadata.loader_preprocessing,
            "normalize-features",
        ),
    )
    return LoadedDataset(data=data, metadata=metadata)


__all__ = ["available_datasets", "load_dataset"]
