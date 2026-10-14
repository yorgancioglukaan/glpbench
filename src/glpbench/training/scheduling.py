"""Learning-rate scheduler construction for link-prediction training."""

import math

import torch


_SCHEDULER_TYPES = ("none", "reduce-on-plateau", "cosine-annealing")


def available_lr_schedulers():
    """Return registered learning-rate scheduler names."""
    return _SCHEDULER_TYPES


def validate_scheduler_config(scheduler_type="none", parameters=None, *, epochs):
    """Return a validated canonical scheduler configuration."""
    if scheduler_type not in _SCHEDULER_TYPES:
        raise ValueError(f"Unknown learning-rate scheduler '{scheduler_type}'")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError("lr_scheduler_kwargs must be a dictionary")
    parameters = dict(parameters)
    if scheduler_type == "none":
        if parameters:
            raise ValueError("scheduler 'none' does not accept parameters")
        return {}

    common = {"minimum_learning_rate"}
    allowed = (
        common | {"factor", "patience", "threshold", "cooldown"}
        if scheduler_type == "reduce-on-plateau" else common
    )
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise ValueError(
            f"Unknown {scheduler_type} scheduler parameter: {sorted(unknown)[0]}"
        )

    minimum = parameters.get("minimum_learning_rate", 0.0)
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or not math.isfinite(minimum)
        or minimum < 0
    ):
        raise ValueError("minimum_learning_rate must be non-negative")
    canonical = {"minimum_learning_rate": float(minimum)}
    if scheduler_type == "cosine-annealing":
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("epochs must be positive for cosine annealing")
        return canonical

    factor = parameters.get("factor", 0.1)
    patience = parameters.get("patience", 10)
    threshold = parameters.get("threshold", 1e-4)
    cooldown = parameters.get("cooldown", 0)
    if (
        not isinstance(factor, (int, float))
        or isinstance(factor, bool)
        or not math.isfinite(factor)
        or not 0 < factor < 1
    ):
        raise ValueError("scheduler factor must be between zero and one")
    for name, value in (("patience", patience), ("cooldown", cooldown)):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"scheduler {name} must be a non-negative integer")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("scheduler threshold must be non-negative")
    return {
        "factor": float(factor),
        "patience": patience,
        "threshold": float(threshold),
        "cooldown": cooldown,
        **canonical,
    }


def build_lr_scheduler(optimizer, scheduler_type="none", parameters=None, *, epochs):
    """Construct a scheduler for an optimizer, or return ``None``."""
    if optimizer is None:
        if scheduler_type != "none":
            raise ValueError("learning-rate schedulers require a trainable model")
        return None
    config = validate_scheduler_config(
        scheduler_type, parameters, epochs=epochs
    )
    if scheduler_type == "none":
        return None
    initial_learning_rate = min(
        float(group["lr"]) for group in optimizer.param_groups
    )
    if config["minimum_learning_rate"] > initial_learning_rate:
        raise ValueError(
            "minimum_learning_rate cannot exceed the initial learning rate"
        )
    if scheduler_type == "reduce-on-plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config["factor"],
            patience=config["patience"],
            threshold=config["threshold"],
            cooldown=config["cooldown"],
            min_lr=config["minimum_learning_rate"],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config["minimum_learning_rate"],
    )


def step_lr_scheduler(scheduler, scheduler_type, validation_score):
    """Advance a scheduler after one completed validation evaluation."""
    if scheduler is None:
        return
    if scheduler_type == "reduce-on-plateau":
        scheduler.step(validation_score)
    else:
        scheduler.step()


def current_learning_rate(optimizer):
    """Return the effective learning rate of the first parameter group."""
    if optimizer is None:
        return None
    return float(optimizer.param_groups[0]["lr"])
