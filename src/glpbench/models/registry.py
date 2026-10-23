"""Public registration and lookup for link-prediction model adapters."""

from __future__ import annotations

from collections.abc import Iterable

from glpbench.contracts import ModelCapabilities
from glpbench.exceptions import CompatibilityError
from glpbench.interfaces import ModelAdapter
from glpbench.models.adapters import (
    AGAEAdapter,
    AGAEMLPAdapter,
    DEALAdapter,
    FeatureCosineAdapter,
    FeatureMLPAdapter,
    GAEAdapter,
    GAEMLPAdapter,
    VGAEAdapter,
)


_MODEL_ADAPTERS = {}
_NAME_LOOKUP = {}


def normalize_model_name(name):
    """Return the case- and separator-insensitive form of a model name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    normalized = "".join(
        character for character in name.strip().lower()
        if character.isalnum()
    )
    if not normalized:
        raise ValueError("model name must contain an alphanumeric character")
    return normalized


def register_model_adapter(adapter, *, aliases=(), replace=False):
    """Register a model adapter under its canonical name."""
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    if not isinstance(adapter.capabilities, ModelCapabilities):
        raise TypeError("adapter capabilities must be a ModelCapabilities instance")
    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Iterable):
        raise TypeError("aliases must be an iterable of strings")

    canonical_name = adapter.name.strip().lower()
    canonical_key = normalize_model_name(adapter.name)
    lookup_keys = {
        canonical_key,
        *(normalize_model_name(alias) for alias in tuple(aliases)),
    }
    existing = _MODEL_ADAPTERS.get(canonical_name)
    if existing is not None and not replace:
        raise ValueError(f"model adapter '{canonical_name}' is already registered")
    if existing is None and canonical_key in _NAME_LOOKUP:
        raise ValueError(f"model name '{adapter.name}' collides with an existing alias")
    for key in lookup_keys:
        owner = _NAME_LOOKUP.get(key)
        if owner is not None and owner != canonical_name:
            raise ValueError(
                f"model alias collides with registered model '{owner}'"
            )

    adapters = dict(_MODEL_ADAPTERS)
    lookup = {
        key: owner for key, owner in _NAME_LOOKUP.items()
        if owner != canonical_name
    }
    adapters[canonical_name] = adapter
    lookup.update({key: canonical_name for key in lookup_keys})
    _MODEL_ADAPTERS.clear()
    _MODEL_ADAPTERS.update(adapters)
    _NAME_LOOKUP.clear()
    _NAME_LOOKUP.update(lookup)
    return adapter


def get_model_adapter(name):
    """Return the adapter registered for a model name or alias."""
    canonical_name = _NAME_LOOKUP.get(normalize_model_name(name))
    if canonical_name is None:
        choices = ", ".join(available_model_types())
        raise CompatibilityError(
            f"Unknown model '{name}'. Available models: {choices}.",
            component="model-registry",
            model=str(name),
        )
    return _MODEL_ADAPTERS[canonical_name]


def available_model_types():
    """Return registered canonical model names in stable order."""
    return tuple(sorted(_MODEL_ADAPTERS))


def _register_builtins():
    for adapter_class in (
        GAEAdapter,
        GAEMLPAdapter,
        AGAEAdapter,
        AGAEMLPAdapter,
        VGAEAdapter,
        DEALAdapter,
        FeatureCosineAdapter,
        FeatureMLPAdapter,
    ):
        register_model_adapter(adapter_class(), replace=True)


_register_builtins()


__all__ = [
    "available_model_types",
    "get_model_adapter",
    "normalize_model_name",
    "register_model_adapter",
]
