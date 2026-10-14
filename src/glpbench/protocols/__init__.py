"""Transductive and semi-inductive evaluation protocols."""

from glpbench.protocols.splitting import (
    ComparableSplits,
    InductiveSplits,
    QueryNodeSets,
    ValidationSplits,
    audit_comparable_splits,
    calculate_reduced_features,
    generate_comparable_splits,
    generate_validation_splits,
    inductive_split,
    select_query_node_sets,
    select_query_nodes,
    update_reduced_features,
)

__all__ = [
    "ComparableSplits",
    "InductiveSplits",
    "QueryNodeSets",
    "ValidationSplits",
    "audit_comparable_splits",
    "calculate_reduced_features",
    "generate_comparable_splits",
    "generate_validation_splits",
    "inductive_split",
    "select_query_node_sets",
    "select_query_nodes",
    "update_reduced_features",
]
