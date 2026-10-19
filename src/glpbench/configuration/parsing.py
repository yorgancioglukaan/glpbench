"""Strict, path-aware primitives for parsing configuration documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

from glpbench.exceptions import ConfigurationError


PathPart = str | int
FieldPath = tuple[PathPart, ...]
T = TypeVar("T")


def render_path(path: FieldPath) -> str:
    """Render a field path using dotted names and sequence indexes."""
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + part
    return rendered or "<document>"


def configuration_error(path: FieldPath, message: str) -> ConfigurationError:
    """Create a path-aware configuration exception."""
    field = render_path(path)
    return ConfigurationError(
        f"{field} {message}",
        component="configuration-parser",
        context={"field_path": field},
    )


def require_mapping(value: Any, path: FieldPath) -> Mapping[str, Any]:
    """Require a mapping with string keys."""
    if not isinstance(value, Mapping):
        raise configuration_error(path, "must be a mapping")
    for key in value:
        if not isinstance(key, str) or not key:
            raise configuration_error(path, "must use non-empty string keys")
    return value


def reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: set[str] | frozenset[str], path: FieldPath
) -> None:
    """Reject keys outside an explicitly allowed set."""
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise configuration_error(path + (unknown[0],), "is not a recognized field")


def require_keys(
    mapping: Mapping[str, Any], required: set[str] | frozenset[str], path: FieldPath
) -> None:
    """Require every key in an explicitly required set."""
    missing = sorted(required.difference(mapping))
    if missing:
        raise configuration_error(path + (missing[0],), "is required")


def require_string(value: Any, path: FieldPath) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise configuration_error(path, "must be a non-empty string")
    return value


def require_boolean(value: Any, path: FieldPath) -> bool:
    """Require a Boolean value."""
    if not isinstance(value, bool):
        raise configuration_error(path, "must be a boolean")
    return value


def require_integer(
    value: Any,
    path: FieldPath,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Require an integer within optional inclusive bounds."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise configuration_error(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise configuration_error(path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise configuration_error(path, f"must be at most {maximum}")
    return value


def require_number(
    value: Any,
    path: FieldPath,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    """Require a finite numeric value within optional bounds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise configuration_error(path, "must be a number")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise configuration_error(path, "must be finite")
    if minimum is not None and (
        number < minimum or (minimum_exclusive and number == minimum)
    ):
        relation = "greater than" if minimum_exclusive else "at least"
        raise configuration_error(path, f"must be {relation} {minimum}")
    if maximum is not None and (
        number > maximum or (maximum_exclusive and number == maximum)
    ):
        relation = "less than" if maximum_exclusive else "at most"
        raise configuration_error(path, f"must be {relation} {maximum}")
    return number


def require_sequence(value: Any, path: FieldPath) -> Sequence[Any]:
    """Require a non-string sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise configuration_error(path, "must be a sequence")
    return value


def require_unique(values: Sequence[T], path: FieldPath) -> tuple[T, ...]:
    """Require a non-empty sequence whose values compare uniquely."""
    if not values:
        raise configuration_error(path, "must not be empty")
    for index, value in enumerate(values):
        if value in values[:index]:
            raise configuration_error(path + (index,), "duplicates an earlier value")
    return tuple(values)


def optional(
    mapping: Mapping[str, Any], key: str, parser, path: FieldPath, default=None
):
    """Parse an optional mapping field or return its declared default."""
    if key not in mapping:
        return default
    return parser(mapping[key], path + (key,))


def load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    """Load a YAML file and require a mapping document root."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigurationError(
            f"could not read configuration file: {source}",
            component="configuration-parser",
            context={"path": str(source)},
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"invalid YAML in configuration file: {source}",
            component="configuration-parser",
            context={"path": str(source)},
        ) from error
    return require_mapping(document, ())


__all__ = [
    "FieldPath",
    "configuration_error",
    "load_yaml_mapping",
    "optional",
    "reject_unknown_keys",
    "render_path",
    "require_boolean",
    "require_integer",
    "require_keys",
    "require_mapping",
    "require_number",
    "require_sequence",
    "require_string",
    "require_unique",
]
