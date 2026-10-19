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


def load_config(path: str | Path) -> CanonicalConfig:
    """Load, strictly parse, and semantically validate a canonical YAML file."""
    document = load_yaml_mapping(path)
    kind = document.get("kind")
    if not isinstance(kind, str) or kind not in _PARSERS:
        raise configuration_error(
            ("kind",), f"must be one of {', '.join(sorted(_PARSERS))}"
        )
    return validate_config(_PARSERS[kind](document))


def _load_kind(path, expected_type, expected_kind):
    config = load_config(path)
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
