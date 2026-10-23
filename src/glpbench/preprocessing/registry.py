"""Registration and construction of feature preprocessors."""

from __future__ import annotations

from collections.abc import Iterable

from glpbench.interfaces import FeaturePreprocessor
from glpbench.preprocessing.adapters import (
    IdentityPreprocessor,
    NormalizeFeaturesPreprocessor,
    ReductionPreprocessor,
)


_PREPROCESSOR_FACTORIES = {}
_NAME_LOOKUP = {}


def normalize_preprocessor_name(name):
    """Return the case- and separator-insensitive form of a name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("preprocessor name must be a non-empty string")
    normalized = "".join(
        character for character in name.strip().lower()
        if character.isalnum()
    )
    if not normalized:
        raise ValueError("preprocessor name must contain an alphanumeric character")
    return normalized


def register_preprocessor(name, factory, *, aliases=(), replace=False):
    """Register a preprocessor factory under a stable name."""
    canonical_name = name.strip().lower() if isinstance(name, str) else name
    canonical_key = normalize_preprocessor_name(name)
    if not callable(factory):
        raise TypeError("preprocessor factory must be callable")
    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Iterable):
        raise TypeError("aliases must be an iterable of strings")
    lookup_keys = {canonical_key, *(
        normalize_preprocessor_name(alias) for alias in tuple(aliases)
    )}
    if canonical_name in _PREPROCESSOR_FACTORIES and not replace:
        raise ValueError(f"preprocessor '{canonical_name}' is already registered")
    for key in lookup_keys:
        owner = _NAME_LOOKUP.get(key)
        if owner is not None and owner != canonical_name:
            raise ValueError(
                f"preprocessor alias collides with registered preprocessor '{owner}'"
            )
    factories = dict(_PREPROCESSOR_FACTORIES)
    lookup = {
        key: owner for key, owner in _NAME_LOOKUP.items()
        if owner != canonical_name
    }
    factories[canonical_name] = factory
    lookup.update({key: canonical_name for key in lookup_keys})
    _PREPROCESSOR_FACTORIES.clear()
    _PREPROCESSOR_FACTORIES.update(factories)
    _NAME_LOOKUP.clear()
    _NAME_LOOKUP.update(lookup)
    return factory


def create_preprocessor(name, **parameters):
    """Construct a fresh registered preprocessor."""
    canonical_name = _NAME_LOOKUP.get(normalize_preprocessor_name(name))
    if canonical_name is None:
        choices = ", ".join(available_preprocessors())
        raise ValueError(
            f"Unknown preprocessor '{name}'. Available preprocessors: {choices}."
        )
    preprocessor = _PREPROCESSOR_FACTORIES[canonical_name](**parameters)
    if not isinstance(preprocessor, FeaturePreprocessor):
        raise TypeError("preprocessor factory must return a FeaturePreprocessor")
    return preprocessor


def available_preprocessors():
    """Return registered preprocessor names in stable order."""
    return tuple(sorted(_PREPROCESSOR_FACTORIES))


def _reducer_factory(method):
    return lambda n_components, **parameters: ReductionPreprocessor(
        method=method,
        n_components=n_components,
        parameters=parameters,
    )


def _register_builtins():
    register_preprocessor("identity", IdentityPreprocessor, replace=True)
    register_preprocessor(
        "normalize-features",
        NormalizeFeaturesPreprocessor,
        aliases=("normalize",),
        replace=True,
    )
    for method in ("pca", "truncated-svd", "sparse-random-projection"):
        register_preprocessor(method, _reducer_factory(method), replace=True)


_register_builtins()


__all__ = [
    "available_preprocessors",
    "create_preprocessor",
    "normalize_preprocessor_name",
    "register_preprocessor",
]
