"""Configured adapters around GLPBench split implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from glpbench.contracts import (
    CandidateOrientation,
    EndpointMode,
    ProtocolRequirements,
)
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
    requirements: ClassVar[ProtocolRequirements] = ProtocolRequirements(
        endpoint_modes=(
            EndpointMode.VISIBLE_VISIBLE,
            EndpointMode.VISIBLE_UNSEEN,
        ),
        candidate_orientations=(
            CandidateOrientation.UNDIRECTED,
            CandidateOrientation.VISIBLE_SOURCE_QUERY_TARGET,
        ),
    )
    query: Any = None
    test_ratio: float | None = None
    train_ratio: float = 0.8
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query: Any = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None

    def split(self, data, seed, *, dataset_fingerprint=None):
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
            dataset_fingerprint=dataset_fingerprint,
        )

    def audit(self, data, splits):
        """Audit matched split invariants."""
        return audit_comparable_splits(data, splits)


@dataclass(frozen=True)
class MatchedValidationProtocol:
    """Configure validation-only matched semi-inductive evaluation."""

    name: ClassVar[str] = "matched-semi-inductive-validation-only"
    requirements: ClassVar[ProtocolRequirements] = ProtocolRequirements(
        endpoint_modes=(EndpointMode.VISIBLE_UNSEEN,),
        candidate_orientations=(
            CandidateOrientation.VISIBLE_SOURCE_QUERY_TARGET,
        ),
    )
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None
    train_ratio: float = 0.8

    def split(self, data, seed, *, dataset_fingerprint=None):
        """Delegate validation-only construction to the procedural API."""
        return generate_validation_splits(
            data,
            query_size=self.query_size,
            query_fraction=self.query_fraction,
            validation_query_size=self.validation_query_size,
            validation_query_fraction=self.validation_query_fraction,
            train_ratio=self.train_ratio,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
        )

    def audit(self, data, splits):
        """Audit validation-only split invariants."""
        return audit_validation_splits(data, splits)


@dataclass(frozen=True)
class SemiInductiveProtocol:
    """Configure standalone semi-inductive evaluation."""

    name: ClassVar[str] = "semi-inductive"
    requirements: ClassVar[ProtocolRequirements] = ProtocolRequirements(
        endpoint_modes=(EndpointMode.VISIBLE_UNSEEN,),
        candidate_orientations=(
            CandidateOrientation.VISIBLE_SOURCE_QUERY_TARGET,
        ),
    )
    query: Any = None
    test_ratio: float | None = None
    train_ratio: float = 0.8
    query_size: int | None = None
    query_fraction: float | None = None
    validation_query: Any = None
    validation_query_size: int | None = None
    validation_query_fraction: float | None = None

    def split(self, data, seed, *, dataset_fingerprint=None):
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
            dataset_fingerprint=dataset_fingerprint,
        )

    def audit(self, data, splits):
        """Audit standalone semi-inductive split invariants."""
        return audit_semi_inductive_splits(data, splits)


@dataclass(frozen=True)
class TransductiveProtocol:
    """Configure standalone random-edge transductive evaluation."""

    name: ClassVar[str] = "transductive"
    requirements: ClassVar[ProtocolRequirements] = ProtocolRequirements(
        endpoint_modes=(EndpointMode.VISIBLE_VISIBLE,),
        candidate_orientations=(CandidateOrientation.UNDIRECTED,),
    )
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

    def split(self, data, seed, *, dataset_fingerprint=None):
        """Construct transductive train, validation, and test partitions."""
        return generate_transductive_splits(
            data,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
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
