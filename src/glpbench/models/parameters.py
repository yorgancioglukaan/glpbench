"""Model-agnostic parameter resolution for registered adapters."""

from __future__ import annotations

from collections.abc import Mapping

from glpbench.interfaces import ModelAdapter


def resolve_model_parameters(adapter, supplied):
    """Select, default, and validate parameters declared by an adapter.

    Bundled adapters expose a ``defaults`` mapping that defines their accepted
    construction parameters. Adapters without that optional declaration
    receive the complete supplied mapping and remain responsible for rejecting
    unsupported fields in ``validate_parameters``.
    """
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    if not isinstance(supplied, Mapping):
        raise TypeError("supplied model parameters must be a mapping")

    defaults = getattr(adapter, "defaults", None)
    if isinstance(defaults, Mapping):
        selected = {
            name: supplied[name] for name in defaults if name in supplied
        }
    else:
        selected = dict(supplied)
    return adapter.validate_parameters(selected)


__all__ = ["resolve_model_parameters"]
