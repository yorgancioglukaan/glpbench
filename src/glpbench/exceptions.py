"""Exception hierarchy for GLPBench component failures."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


class GLPBenchError(Exception):
    """Base exception for expected GLPBench failures.

    Optional context identifies the experiment component that failed without
    requiring callers to parse the human-readable message.
    """

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        dataset: str | None = None,
        model: str | None = None,
        seed: int | None = None,
        configuration_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        values = dict(context or {})
        values.update({
            "component": component,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "configuration_id": configuration_id,
        })
        self.message = message
        self.context = MappingProxyType(
            {key: value for key, value in values.items() if value is not None}
        )
        suffix = ", ".join(
            f"{key}={value!r}" for key, value in self.context.items()
        )
        rendered = f"{message} [{suffix}]" if suffix else message
        super().__init__(rendered)


class ConfigurationError(GLPBenchError):
    """Raised when an experiment configuration is invalid."""


class DatasetError(GLPBenchError):
    """Raised when dataset loading or validation fails."""


class ProtocolError(GLPBenchError):
    """Raised when split construction or protocol auditing fails."""


class CompatibilityError(GLPBenchError):
    """Raised when selected components cannot be used together."""


class TrainingError(GLPBenchError):
    """Raised when model optimization or checkpoint selection fails."""


class PersistenceError(GLPBenchError):
    """Raised when reading or writing evaluation artifacts fails."""


__all__ = [
    "CompatibilityError",
    "ConfigurationError",
    "DatasetError",
    "GLPBenchError",
    "PersistenceError",
    "ProtocolError",
    "TrainingError",
]
