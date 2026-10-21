"""Configured adapters around GLPBench split implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from glpbench.protocols.splitting import (
    audit_comparable_splits,
    audit_semi_inductive_splits,
    audit_transductive_splits,
    audit_validation_splits,
    generate_comparable_splits,
    generate_transductive_splits,
    generate_validation_splits,
    inductive_split,
)


@dataclass(frozen=True)
class MatchedSemiInductiveProtocol:
    """Configure matched transductive and semi-inductive evaluation."""

    name: ClassVar[str] = "matched-semi-inductive"
    query: Any = None
    test_ratio: float | None = None
    train_ratio: float = 0.8
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query: Any = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None

    def split(self, data, seed):
        """Delegate matched split construction to the procedural API."""
        return generate_comparable_splits(
            data,
            query=self.query,
            test_ratio=self.test_ratio,
            train_ratio=self.train_ratio,
            query_size=self.query_size,
            query_fraction=self.query_fraction,
            validation_query=self.validation_query,
            validation_query_size=self.validation_query_size,
            validation_query_fraction=self.validation_query_fraction,
            seed=seed,
        )

    def audit(self, data, splits):
        """Audit matched split invariants."""
        return audit_comparable_splits(data, splits)


@dataclass(frozen=True)
class MatchedValidationProtocol:
    """Configure validation-only matched semi-inductive evaluation."""

    name: ClassVar[str] = "matched-semi-inductive-validation-only"
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None
    train_ratio: float = 0.8

    def split(self, data, seed):
        """Delegate validation-only construction to the procedural API."""
        return generate_validation_splits(
            data,
            query_size=self.query_size,
            query_fraction=self.query_fraction,
            validation_query_size=self.validation_query_size,
            validation_query_fraction=self.validation_query_fraction,
            train_ratio=self.train_ratio,
            seed=seed,
        )

    def audit(self, data, splits):
        """Audit validation-only split invariants."""
        return audit_validation_splits(data, splits)


@dataclass(frozen=True)
class SemiInductiveProtocol:
    """Configure standalone semi-inductive evaluation."""

    name: ClassVar[str] = "semi-inductive"
    query: Any = None
    test_ratio: float | None = None
    train_ratio: float = 0.8
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query: Any = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None

    def split(self, data, seed):
        """Delegate semi-inductive construction to the procedural API."""
        return inductive_split(
            data,
            query=self.query,
            test_ratio=self.test_ratio,
            train_ratio=self.train_ratio,
            query_size=self.query_size,
            query_fraction=self.query_fraction,
            validation_query=self.validation_query,
            validation_query_size=self.validation_query_size,
            validation_query_fraction=self.validation_query_fraction,
            seed=seed,
        )

    def audit(self, data, splits):
        """Audit standalone semi-inductive split invariants."""
        return audit_semi_inductive_splits(data, splits)


@dataclass(frozen=True)
class TransductiveProtocol:
    """Configure standalone random-edge transductive evaluation."""

    name: ClassVar[str] = "transductive"
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

    def split(self, data, seed):
        """Construct transductive train, validation, and test partitions."""
        return generate_transductive_splits(
            data,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            seed=seed,
        )

    def audit(self, data, splits):
        """Audit standalone transductive split invariants."""
        return audit_transductive_splits(data, splits)


__all__ = [
    "MatchedSemiInductiveProtocol",
    "MatchedValidationProtocol",
    "SemiInductiveProtocol",
    "TransductiveProtocol",
]
