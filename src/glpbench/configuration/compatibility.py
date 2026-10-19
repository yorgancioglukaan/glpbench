"""Compatibility dispatch for configuration formats predating canonical schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from glpbench.configuration.parsing import configuration_error


_LEGACY_KINDS = {
    "benchmark",
    "selected-hyperparameter-benchmark",
    "hyperparameter-search",
    "training-policy-ablation",
    "validation-convergence-audit",
    "selected-hyperparameters",
}


def identify_legacy_kind(document: Mapping[str, Any]) -> str:
    """Identify the command or artifact represented by a legacy document."""
    kind = document.get("kind", "benchmark")
    if not isinstance(kind, str) or kind not in _LEGACY_KINDS:
        raise configuration_error(
            ("kind",), f"references unsupported legacy kind {kind!r}"
        )
    return kind


def load_legacy_config(path: str | Path, document: Mapping[str, Any]):
    """Load a legacy document through its behavior-preserving parser."""
    if document.get("version") not in {1, 2}:
        raise configuration_error(
            ("version",), "must identify a supported legacy schema version"
        )
    kind = identify_legacy_kind(document)
    if kind in {
        "benchmark", "selected-hyperparameter-benchmark",
        "training-policy-ablation",
    }:
        from glpbench.runners.benchmark import load_benchmark_config

        return load_benchmark_config(path)
    if kind == "hyperparameter-search":
        from glpbench.runners.search import load_search_config

        return load_search_config(path)
    if kind == "validation-convergence-audit":
        from glpbench.runners.audit import load_audit_config

        return load_audit_config(path)

    from glpbench.runners.benchmark import _load_selected_configurations

    return _load_selected_configurations(Path(path))


__all__ = ["identify_legacy_kind", "load_legacy_config"]
