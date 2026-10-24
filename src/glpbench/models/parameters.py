"""Model-agnostic parameter resolution for registered adapters."""

from __future__ import annotations

from collections.abc import Mapping

from glpbench.interfaces import ModelAdapter


def resolve_model_parameters(adapter, supplied):
    """Select, default, and validate parameters declared by an adapter.

    Adapters expose ``parameter_defaults`` to define their accepted
    construction parameters. The selected values are still passed through the
    adapter's authoritative value validation.
    """
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    if not isinstance(supplied, Mapping):
        raise TypeError("supplied model parameters must be a mapping")

    defaults = adapter.parameter_defaults
    if not isinstance(defaults, Mapping):
        raise TypeError("adapter parameter_defaults must be a mapping")
    selected = {
        name: supplied[name] for name in defaults if name in supplied
    }
    return adapter.validate_parameters(selected)


__all__ = ["resolve_model_parameters"]
