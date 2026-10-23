"""Resolve registered evaluation components at runner boundaries."""

from __future__ import annotations

from glpbench.contracts import AuditReport, LoadedDataset, SplitBundle
from glpbench.datasets import load_dataset
from glpbench.exceptions import ProtocolError
from glpbench.protocols import create_split_protocol
from glpbench.protocols import get_split_protocol_requirements
from glpbench.models import get_model_registration
from glpbench.evaluation.compatibility import (
    validate_model_protocol_compatibility,
)


def load_registered_dataset(
    name, *, root="data", normalize_features=False, device="cpu"
):
    """Load a dataset through the registered-provider façade."""
    loaded = load_dataset(
        name,
        root=root,
        normalize_features=normalize_features,
        device=device,
    )
    if not isinstance(loaded, LoadedDataset):
        raise TypeError("dataset resolution must return LoadedDataset")
    return loaded


def realize_registered_split(
    protocol_name,
    data,
    *,
    seed,
    dataset_fingerprint=None,
    parameters=None,
):
    """Construct and audit a split through its registered protocol."""
    protocol = create_split_protocol(protocol_name, **dict(parameters or {}))
    splits = protocol.split(
        data,
        seed,
        dataset_fingerprint=dataset_fingerprint,
    )
    if not isinstance(splits, SplitBundle):
        raise ProtocolError(
            "split protocol must return a SplitBundle",
            component=protocol_name,
            seed=seed,
        )
    report = protocol.audit(data, splits)
    if not isinstance(report, AuditReport):
        raise ProtocolError(
            "split protocol audit must return an AuditReport",
            component=protocol_name,
            seed=seed,
        )
    if not report.passed:
        failed = ", ".join(
            name for name, passed in report.checks.items() if not passed
        )
        message = "split protocol audit failed"
        if failed:
            message += f": {failed}"
        raise ProtocolError(message, component=protocol_name, seed=seed)
    return splits


def validate_registered_compatibility(
    model_name, protocol_name, dataset_metadata, *, device
):
    """Validate one registered model, protocol, dataset, and device choice."""
    registration = get_model_registration(model_name)
    requirements = get_split_protocol_requirements(protocol_name)
    validate_model_protocol_compatibility(
        registration.capabilities,
        requirements,
        dataset_metadata,
        device=device,
        model_name=model_name,
        protocol_name=protocol_name,
    )


__all__ = [
    "load_registered_dataset",
    "realize_registered_split",
    "validate_registered_compatibility",
]
