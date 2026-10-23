"""Ordered execution of registered feature preprocessors."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from glpbench.preprocessing.registry import create_preprocessor


@dataclass(frozen=True)
class PreprocessingStep:
    """Describe one registered preprocessing operation."""

    name: str
    parameters: dict | None = None


@dataclass(frozen=True)
class FeaturePreparation:
    """Contain encoder features and optional reduced features."""

    features: torch.Tensor
    reduced_features: torch.Tensor | None
    preprocessors: tuple[object, ...]
    fit_time_seconds: float | None


class PreprocessingPipeline:
    """Fit and apply an ordered sequence of registered preprocessors."""

    def __init__(self, steps):
        self.steps = tuple(steps)
        if not self.steps:
            raise ValueError("preprocessing pipeline requires at least one step")
        if any(not isinstance(step, PreprocessingStep) for step in self.steps):
            raise TypeError("steps must contain PreprocessingStep objects")

    def fit_transform(self, features, fit_mask, *, seed=None):
        """Fit applicable stages on ``fit_mask`` and transform all rows."""
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a two-dimensional tensor")
        mask = torch.as_tensor(fit_mask, dtype=torch.bool, device=features.device)
        if mask.ndim != 1 or mask.numel() != features.size(0):
            raise ValueError("fit_mask must have one entry per feature row")
        if not bool(mask.any()):
            raise ValueError("fit_mask must select at least one row")

        current = features.clone()
        encoder_features = current
        reduced_features = None
        fitted = []
        fit_time = 0.0
        fitted_any = False
        saw_reduction = False
        for index, step in enumerate(self.steps):
            preprocessor = create_preprocessor(
                step.name, **dict(step.parameters or {})
            )
            output = getattr(preprocessor, "output", "features")
            if output not in {"features", "reduced_features"}:
                raise ValueError("preprocessor output must be features or reduced_features")
            if saw_reduction:
                raise ValueError("dimensionality reduction must be the final pipeline step")
            if output == "reduced_features" and index != len(self.steps) - 1:
                raise ValueError("dimensionality reduction must be the final pipeline step")

            start = time.perf_counter()
            preprocessor.fit(current, mask, seed=seed)
            transformed = preprocessor.transform(current)
            elapsed = time.perf_counter() - start
            if preprocessor.requires_fitting:
                fitted_any = True
                fit_time += elapsed
            if output == "features":
                encoder_features = transformed
                current = transformed
            else:
                saw_reduction = True
                reduced_features = transformed
                current = transformed
            fitted.append(preprocessor)

        return FeaturePreparation(
            features=encoder_features,
            reduced_features=reduced_features,
            preprocessors=tuple(fitted),
            fit_time_seconds=fit_time if fitted_any else None,
        )


__all__ = [
    "FeaturePreparation",
    "PreprocessingPipeline",
    "PreprocessingStep",
]
