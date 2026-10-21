"""Registration and construction of graph-splitting protocols."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from glpbench.exceptions import ProtocolError
from glpbench.interfaces import SplitProtocol


_PROTOCOL_FACTORIES = {}
_NAME_LOOKUP = {}


def normalize_protocol_name(name):
    """Return the case- and separator-insensitive form of a protocol name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("protocol name must be a non-empty string")
    normalized = "".join(
        character for character in name.strip().lower()
        if character.isalnum()
    )
    if not normalized:
        raise ValueError("protocol name must contain an alphanumeric character")
    return normalized


def register_split_protocol(name, factory, *, aliases=(), replace=False):
    """Register a configured-protocol factory under a stable name."""
    canonical_name = name.strip().lower() if isinstance(name, str) else name
    canonical_key = normalize_protocol_name(name)
    if not callable(factory):
        raise TypeError("protocol factory must be callable")
    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Iterable):
        raise TypeError("aliases must be an iterable of strings")
    lookup_keys = {canonical_key}
    for alias in tuple(aliases):
        lookup_keys.add(normalize_protocol_name(alias))

    existing = _PROTOCOL_FACTORIES.get(canonical_name)
    if existing is not None and not replace:
        raise ValueError(f"split protocol '{canonical_name}' is already registered")
    if existing is None and canonical_key in _NAME_LOOKUP:
        raise ValueError(f"protocol name '{name}' collides with an existing alias")
    for key in lookup_keys:
        owner = _NAME_LOOKUP.get(key)
        if owner is not None and owner != canonical_name:
            raise ValueError(
                f"protocol alias collides with registered protocol '{owner}'"
            )

    factories = dict(_PROTOCOL_FACTORIES)
    lookup = {
        key: owner for key, owner in _NAME_LOOKUP.items()
        if owner != canonical_name
    }
    factories[canonical_name] = factory
    lookup.update({key: canonical_name for key in lookup_keys})
    _PROTOCOL_FACTORIES.clear()
    _PROTOCOL_FACTORIES.update(factories)
    _NAME_LOOKUP.clear()
    _NAME_LOOKUP.update(lookup)
    return factory


def get_split_protocol_factory(name):
    """Return the factory registered for a protocol name or alias."""
    canonical_name = _NAME_LOOKUP.get(normalize_protocol_name(name))
    if canonical_name is None:
        choices = ", ".join(available_split_protocols())
        raise ProtocolError(
            f"Unknown split protocol '{name}'. Available protocols: {choices}.",
            component="protocol-registry",
        )
    return _PROTOCOL_FACTORIES[canonical_name]


def create_split_protocol(name, **parameters):
    """Construct and validate a configured split protocol."""
    protocol = get_split_protocol_factory(name)(**parameters)
    if not isinstance(protocol, SplitProtocol):
        raise TypeError("protocol factory must return a SplitProtocol")
    if normalize_protocol_name(protocol.name) != normalize_protocol_name(name):
        canonical_name = _NAME_LOOKUP[normalize_protocol_name(name)]
        if normalize_protocol_name(protocol.name) != normalize_protocol_name(
            canonical_name
        ):
            raise ValueError(
                "constructed protocol name does not match its registry entry"
            )
    return protocol


def available_split_protocols():
    """Return registered split-protocol names in stable order."""
    return tuple(sorted(_PROTOCOL_FACTORIES))


__all__ = [
    "available_split_protocols",
    "create_split_protocol",
    "get_split_protocol_factory",
    "register_split_protocol",
]
