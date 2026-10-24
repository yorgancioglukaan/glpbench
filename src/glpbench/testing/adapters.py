"""Reusable conformance checks for model adapters."""

from __future__ import annotations

from collections.abc import Mapping
import torch

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.interfaces import ModelAdapter


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tensor_snapshot(data) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in data
        if isinstance(value, torch.Tensor)
    }


def _value_snapshot(value):
    if isinstance(value, Mapping):
        return {name: _value_snapshot(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_value_snapshot(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _assert_data_unchanged(data, snapshot) -> None:
    current = _tensor_snapshot(data)
    _assert(
        current.keys() == snapshot.keys(),
        "adapter must not add or remove tensor attributes on input data",
    )
    for name, expected in snapshot.items():
        _assert(
            torch.equal(current[name], expected),
            f"adapter mutated data tensor {name!r}",
        )


def assert_model_adapter_conforms(
    adapter,
    data,
    parameters=None,
    *,
    edge_label_index=None,
    edge_label=None,
    criterion=None,
) -> None:
    """Assert that an adapter satisfies GLPBench's model boundary.

    Parameters
    ----------
    adapter
        Model adapter to check.
    data
        Capability-complete toy graph accepted by the adapter.
    parameters
        Optional non-default construction parameters.
    edge_label_index, edge_label
        Candidate edges and binary labels used for the training check. They
        default to the corresponding attributes on ``data``.
    criterion
        Optional binary classification loss. Defaults to
        :class:`torch.nn.BCEWithLogitsLoss`.

    Raises
    ------
    AssertionError
        If the adapter violates the public contract.
    """
    _assert(isinstance(adapter, ModelAdapter), "adapter must implement ModelAdapter")
    _assert(
        isinstance(adapter.name, str) and bool(adapter.name.strip()),
        "adapter name must be a non-empty string",
    )
    _assert(
        isinstance(adapter.capabilities, ModelCapabilities),
        "adapter capabilities must be ModelCapabilities",
    )
    _assert(
        isinstance(adapter.parameter_defaults, Mapping),
        "adapter parameter_defaults must be a mapping",
    )
    try:
        adapter.parameter_defaults["__glpbench_mutation_check__"] = None
    except TypeError:
        pass
    else:
        adapter.parameter_defaults.pop("__glpbench_mutation_check__", None)
        raise AssertionError("adapter parameter_defaults must be immutable")

    supplied = dict(adapter.parameter_defaults if parameters is None else parameters)
    supplied_before = _value_snapshot(supplied)
    normalized = adapter.validate_parameters(supplied)
    _assert(isinstance(normalized, Mapping), "validated parameters must be a mapping")
    _assert(
        _value_snapshot(supplied) == supplied_before,
        "parameter validation mutated its input",
    )
    try:
        normalized["__glpbench_mutation_check__"] = None
    except TypeError:
        pass
    else:
        normalized.pop("__glpbench_mutation_check__", None)
        raise AssertionError("validated parameters must be immutable")
    try:
        adapter.validate_parameters({"__glpbench_undeclared_parameter__": True})
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("adapter must reject undeclared parameters")

    data_before = _tensor_snapshot(data)
    model = adapter.build(data, normalized)
    _assert(isinstance(model, torch.nn.Module), "adapter build() must return nn.Module")
    _assert_data_unchanged(data, data_before)

    candidates = (
        data.edge_label_index if edge_label_index is None else edge_label_index
    )
    labels = data.edge_label if edge_label is None else edge_label
    _assert(
        isinstance(candidates, torch.Tensor)
        and candidates.ndim == 2
        and candidates.size(0) == 2,
        "edge_label_index must have shape [2, num_edges]",
    )
    _assert(
        isinstance(labels, torch.Tensor)
        and labels.numel() == candidates.size(1),
        "edge labels must match the number of candidate edges",
    )

    if adapter.capabilities.trainable:
        loss = adapter.compute_training_loss(
            model,
            data,
            candidates,
            labels,
            criterion or torch.nn.BCEWithLogitsLoss(),
        )
        _assert(isinstance(loss, TrainingLoss), "training hook must return TrainingLoss")
        _assert(loss.total.ndim == 0, "total training loss must be scalar")
        _assert(torch.isfinite(loss.total).item(), "total training loss must be finite")
        _assert(
            loss.total.requires_grad,
            "total training loss must support backpropagation",
        )
        _assert(bool(loss.components), "training loss must expose components")
        for name, component in loss.components.items():
            _assert(
                isinstance(name, str) and bool(name),
                "training-loss component names must be non-empty strings",
            )
            _assert(
                isinstance(component, torch.Tensor) and component.ndim == 0,
                f"training-loss component {name!r} must be a scalar tensor",
            )
            _assert(
                torch.isfinite(component).item(),
                f"training-loss component {name!r} must be finite",
            )
        model.zero_grad(set_to_none=True)
        loss.total.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        _assert(bool(trainable), "trainable adapter built a model without trainable parameters")
        _assert(
            any(parameter.grad is not None for parameter in trainable),
            "training loss did not produce a model-parameter gradient",
        )
    else:
        try:
            adapter.compute_training_loss(
                model,
                data,
                candidates,
                labels,
                criterion or torch.nn.BCEWithLogitsLoss(),
            )
        except Exception:
            pass
        else:
            raise AssertionError("non-trainable adapter accepted a training step")

    scores = adapter.score_edges(model, data, candidates)
    _assert(isinstance(scores, torch.Tensor), "edge scores must be a tensor")
    _assert(scores.ndim == 1, "edge scores must be one-dimensional")
    _assert(
        scores.numel() == candidates.size(1),
        "adapter must return one score per candidate edge",
    )
    _assert(torch.isfinite(scores).all().item(), "edge scores must be finite")
    if adapter.capabilities.output_kind == "probabilities":
        _assert(
            bool(((scores >= 0) & (scores <= 1)).all()),
            "probability scores must lie in [0, 1]",
        )

    metadata_input = dict(normalized)
    metadata_before = _value_snapshot(metadata_input)
    metadata = adapter.result_metadata(metadata_input)
    _assert(isinstance(metadata, Mapping), "result metadata must be a mapping")
    _assert(
        _value_snapshot(metadata_input) == metadata_before,
        "result metadata mutated parameters",
    )
    _assert_data_unchanged(data, data_before)


__all__ = ["assert_model_adapter_conforms"]
