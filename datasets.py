"""Load and standardize benchmark graph datasets."""

from pathlib import Path

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
from torch_geometric.utils import is_undirected


_PLANETOID_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def _load_planetoid(name, root):
    """Load one graph from the Planetoid family."""
    dataset = Planetoid(root=str(root), name=_PLANETOID_NAMES[name])
    return dataset[0].clone(), _PLANETOID_NAMES[name], "Planetoid"


def _load_amazon(name, root):
    """Load an Amazon co-purchase graph."""
    dataset_name = "Photo" if name == "amazon-photo" else "Computers"
    dataset = Amazon(root=str(root), name=dataset_name)
    return dataset[0].clone(), f"Amazon{dataset_name}", "Amazon"


def _load_coauthor(name, root):
    """Load the Coauthor CS graph."""
    dataset = Coauthor(root=str(root), name="CS")
    return dataset[0].clone(), "CoauthorCS", "Coauthor"


def _load_citation_full(name, root):
    """Load the full DBLP citation graph as undirected."""
    dataset = CitationFull(root=str(root), name="DBLP", to_undirected=True)
    return dataset[0].clone(), "DBLP", "CitationFull"


def _load_wikics(name, root):
    """Load WikiCS as an undirected graph."""
    dataset = WikiCS(root=str(root), is_undirected=True)
    return dataset[0].clone(), "WikiCS", "WikiCS"


def _load_deezer(name, root):
    """Load the Deezer Europe graph."""
    dataset = DeezerEurope(root=str(root))
    return dataset[0].clone(), "DeezerEurope", "Deezer"


_DATASET_LOADERS = {
    **{name: _load_planetoid for name in _PLANETOID_NAMES},
    "amazon-photo": _load_amazon,
    "amazon-computers": _load_amazon,
    "coauthor-cs": _load_coauthor,
    "dblp": _load_citation_full,
    "wikics": _load_wikics,
    "deezer-europe": _load_deezer,
}

_DATASET_ALIASES = {
    "".join(character for character in name if character.isalnum()): name
    for name in _DATASET_LOADERS
}


def _normalize_dataset_name(name):
    """Normalize case, spaces, underscores, and hyphens in dataset names."""
    compact_name = "".join(
        character for character in name.strip().lower() if character.isalnum()
    )
    return _DATASET_ALIASES.get(compact_name)


def available_datasets():
    """Return supported dataset names in stable order.

    Returns
    -------
    list of str
        Normalized dataset names.
    """
    return sorted(_DATASET_LOADERS)


def load_dataset(name, root="data", normalize_features=True, device="cpu"):
    """Load and standardize a benchmark graph.

    Parameters
    ----------
    name : str
        Dataset name. Matching is case-insensitive.
    root : str or pathlib.Path, optional
        Directory used for downloaded and cached data.
    normalize_features : bool, optional
        Whether to normalize node features row-wise.
    device : str or torch.device, optional
        Device receiving the returned graph.

    Returns
    -------
    torch_geometric.data.Data
        Cloned graph with dataset metadata attached.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    normalized_name = _normalize_dataset_name(name)
    if normalized_name is None:
        choices = ", ".join(available_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {choices}.")
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

    data, canonical_name, family = _DATASET_LOADERS[normalized_name](
        normalized_name, Path(root)
    )
    if not isinstance(data, Data):
        raise TypeError("dataset loader must return torch_geometric.data.Data")
    if data.x is None:
        raise ValueError("loaded dataset does not contain node features")
    if data.edge_index is None:
        raise ValueError("loaded dataset does not contain edge_index")
    if data.num_nodes is None:
        raise ValueError("loaded dataset does not define num_nodes")
    if not is_undirected(data.edge_index):
        raise ValueError("loaded dataset must be undirected")

    if normalize_features:
        data = NormalizeFeatures()(data)
    data = data.to(target_device)
    data.dataset_name = canonical_name
    data.dataset_family = family
    return data
