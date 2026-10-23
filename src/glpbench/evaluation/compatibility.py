"""Declarative compatibility checks for models and evaluation protocols."""

from __future__ import annotations

import torch

from glpbench.contracts import (
    DatasetMetadata,
    ModelCapabilities,
    ProtocolRequirements,
)
from glpbench.exceptions import CompatibilityError


def _device_type(device):
    """Return the canonical device type used by capability declarations."""
    try:
        return torch.device(device).type
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid device: {device!r}") from error


def validate_model_protocol_compatibility(
    model_capabilities,
    protocol_requirements,
    dataset_metadata,
    *,
    device,
    model_name=None,
    protocol_name=None,
):
    """Raise an actionable error when selected components are incompatible.

    Parameters
    ----------
    model_capabilities : ModelCapabilities
        Inputs and evaluation settings supported by the model.
    protocol_requirements : ProtocolRequirements
        Inputs and endpoint semantics required by the protocol.
    dataset_metadata : DatasetMetadata
        Structural and feature metadata for the loaded graph.
    device : str or torch.device
        Requested execution device.
    model_name, protocol_name : str, optional
        Names included in diagnostic context.
    """
    if not isinstance(model_capabilities, ModelCapabilities):
        raise TypeError("model_capabilities must be ModelCapabilities")
    if not isinstance(protocol_requirements, ProtocolRequirements):
        raise TypeError("protocol_requirements must be ProtocolRequirements")
    if not isinstance(dataset_metadata, DatasetMetadata):
        raise TypeError("dataset_metadata must be DatasetMetadata")

    failures = []
    supported_modes = set(model_capabilities.supported_endpoint_modes)
    missing_modes = [
        mode.value for mode in protocol_requirements.endpoint_modes
        if mode not in supported_modes
    ]
    if missing_modes:
        failures.append(
            "protocol requires unsupported endpoint mode(s): "
            + ", ".join(missing_modes)
        )

    supported_orientations = set(
        model_capabilities.supported_candidate_orientations
    )
    missing_orientations = [
        orientation.value
        for orientation in protocol_requirements.candidate_orientations
        if orientation not in supported_orientations
    ]
    if missing_orientations:
        failures.append(
            "protocol requires unsupported candidate orientation(s): "
            + ", ".join(missing_orientations)
        )

    if model_capabilities.requires_node_features and (
        not protocol_requirements.provides_node_features
        or not dataset_metadata.has_node_features
    ):
        failures.append("model requires node features")
    if model_capabilities.requires_reduced_features and (
        not protocol_requirements.provides_node_features
        or not dataset_metadata.has_node_features
    ):
        failures.append("model requires reducible node features")
    if (
        model_capabilities.uses_message_passing
        or model_capabilities.requires_graph_context
    ) and not protocol_requirements.provides_message_passing_graph:
        failures.append("model requires a message-passing graph")
    if (
        model_capabilities.requires_labels
        and not protocol_requirements.provides_edge_labels
    ):
        failures.append("model requires labeled candidate edges")
    if (
        model_capabilities.requires_auxiliary_data
        and not protocol_requirements.provides_auxiliary_data
    ):
        failures.append("model requires auxiliary protocol data")
    if protocol_requirements.requires_undirected_graph and dataset_metadata.directed:
        failures.append("protocol requires an undirected graph")

    resolved_device = _device_type(device)
    if resolved_device not in model_capabilities.supported_devices:
        supported = ", ".join(model_capabilities.supported_devices)
        failures.append(
            f"device {resolved_device!r} is unsupported; supported devices: "
            f"{supported}"
        )

    if failures:
        model_label = model_name or "<unknown>"
        protocol_label = protocol_name or "<unknown>"
        raise CompatibilityError(
            f"Model {model_label!r} is incompatible with protocol "
            f"{protocol_label!r}: " + "; ".join(failures),
            component="model-protocol-compatibility",
            dataset=dataset_metadata.name,
            model=model_name,
            context={
                "protocol": protocol_name,
                "device": resolved_device,
                "failures": tuple(failures),
            },
        )


__all__ = ["validate_model_protocol_compatibility"]
