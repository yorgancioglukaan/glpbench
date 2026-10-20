"""Registration and lookup of graph dataset providers."""

from __future__ import annotations

from collections.abc import Iterable

from glpbench.exceptions import DatasetError
from glpbench.interfaces import DatasetProvider


_PROVIDERS = {}
_NAME_LOOKUP = {}


def normalize_dataset_name(name):
    """Return the case- and separator-insensitive form of a dataset name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset name must be a non-empty string")
    normalized = "".join(
        character for character in name.strip().lower()
        if character.isalnum()
    )
    if not normalized:
        raise ValueError("dataset name must contain an alphanumeric character")
    return normalized


def register_dataset_provider(provider, *, aliases=(), replace=False):
    """Register a dataset provider and optional lookup aliases."""
    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    if not isinstance(provider, DatasetProvider):
        raise TypeError("provider must implement the DatasetProvider protocol")
    if not isinstance(provider.name, str) or not provider.name.strip():
        raise ValueError("provider.name must be a non-empty string")
    canonical_name = provider.name.strip().lower()
    canonical_key = normalize_dataset_name(canonical_name)
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Iterable):
        raise TypeError("aliases must be an iterable of strings")
    lookup_keys = {canonical_key}
    for alias in tuple(aliases):
        lookup_keys.add(normalize_dataset_name(alias))

    existing_provider = _PROVIDERS.get(canonical_name)
    if existing_provider is not None and not replace:
        raise ValueError(
            f"dataset provider '{canonical_name}' is already registered"
        )
    if existing_provider is None and canonical_key in _NAME_LOOKUP:
        raise ValueError(
            f"dataset name '{provider.name}' collides with an existing alias"
        )
    for key in lookup_keys:
        owner = _NAME_LOOKUP.get(key)
        if owner is not None and owner != canonical_name:
            raise ValueError(
                f"dataset alias collides with registered provider '{owner}'"
            )

    new_providers = dict(_PROVIDERS)
    new_lookup = {
        key: owner for key, owner in _NAME_LOOKUP.items()
        if owner != canonical_name
    }
    new_providers[canonical_name] = provider
    for key in lookup_keys:
        new_lookup[key] = canonical_name
    _PROVIDERS.clear()
    _PROVIDERS.update(new_providers)
    _NAME_LOOKUP.clear()
    _NAME_LOOKUP.update(new_lookup)
    return provider


def get_dataset_provider(name):
    """Return the provider registered for ``name`` or one of its aliases."""
    key = normalize_dataset_name(name)
    canonical_name = _NAME_LOOKUP.get(key)
    if canonical_name is None:
        choices = ", ".join(available_datasets())
        raise DatasetError(
            f"Unknown dataset '{name}'. Available datasets: {choices}.",
            component="dataset-registry",
            dataset=name,
        )
    return _PROVIDERS[canonical_name]


def available_datasets():
    """Return canonical registered dataset names in stable order."""
    return sorted(_PROVIDERS)


__all__ = [
    "available_datasets",
    "get_dataset_provider",
    "register_dataset_provider",
]
