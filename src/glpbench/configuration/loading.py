"""Public loading entry points for canonical configurations."""

from __future__ import annotations

from pathlib import Path

from glpbench.configuration.ablation import AblationConfig, parse_ablation_config
from glpbench.configuration.audit import AuditConfig, parse_audit_config
from glpbench.configuration.benchmark import BenchmarkConfig, parse_benchmark_config
from glpbench.configuration.parsing import configuration_error, load_yaml_mapping
from glpbench.configuration.search import SearchConfig, parse_search_config
from glpbench.configuration.validation import CanonicalConfig, validate_config


_PARSERS = {
    "benchmark": parse_benchmark_config,
    "hyperparameter-search": parse_search_config,
    "training-policy-ablation": parse_ablation_config,
    "validation-convergence-audit": parse_audit_config,
}


def _parse_canonical_document(document) -> CanonicalConfig:
    """Strictly parse and semantically validate a canonical document."""
    kind = document.get("kind")
    if not isinstance(kind, str) or kind not in _PARSERS:
        raise configuration_error(
            ("kind",), f"must be one of {', '.join(sorted(_PARSERS))}"
        )
    return validate_config(_PARSERS[kind](document))


def load_config(path: str | Path):
    """Load a canonical or supported legacy configuration file.

    Canonical documents declare ``schema_version`` and return configuration
    contracts from :mod:`glpbench.configuration`. Legacy documents declare
    ``version`` and return their existing frozen runner contracts so expansion,
    identifiers, defaults, and ordering remain unchanged.
    """
    document = load_yaml_mapping(path)
    has_canonical_version = "schema_version" in document
    has_legacy_version = "version" in document
    if has_canonical_version and has_legacy_version:
        raise configuration_error(
            (), "cannot define both schema_version and legacy version"
        )
    if has_canonical_version:
        return _parse_canonical_document(document)
    if has_legacy_version:
        from glpbench.configuration.compatibility import load_legacy_config

        return load_legacy_config(path, document)
    raise configuration_error(
        (), "must define schema_version or legacy version"
    )


def _load_kind(path, expected_type, expected_kind):
    document = load_yaml_mapping(path)
    if "version" in document:
        raise configuration_error(
            ("version",),
            "uses a legacy format; call load_config for compatibility loading",
        )
    config = _parse_canonical_document(document)
    if not isinstance(config, expected_type):
        raise configuration_error(("kind",), f"must be {expected_kind!r}")
    return config


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load a canonical benchmark configuration."""
    return _load_kind(path, BenchmarkConfig, "benchmark")


def load_search_config(path: str | Path) -> SearchConfig:
    """Load a canonical hyperparameter-search configuration."""
    return _load_kind(path, SearchConfig, "hyperparameter-search")


def load_ablation_config(path: str | Path) -> AblationConfig:
    """Load a canonical training-policy-ablation configuration."""
    return _load_kind(path, AblationConfig, "training-policy-ablation")


def load_audit_config(path: str | Path) -> AuditConfig:
    """Load a canonical validation-convergence-audit configuration."""
    return _load_kind(path, AuditConfig, "validation-convergence-audit")


__all__ = [
    "load_ablation_config", "load_audit_config", "load_benchmark_config",
    "load_config", "load_search_config",
]
