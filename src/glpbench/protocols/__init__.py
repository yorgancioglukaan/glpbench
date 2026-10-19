"""Transductive and semi-inductive evaluation protocols."""

from glpbench.protocols.splitting import (
    QueryNodeSets,
    audit_comparable_splits,
    calculate_reduced_features,
    generate_comparable_splits,
    generate_validation_splits,
    inductive_split,
    select_query_node_sets,
    select_query_nodes,
    update_reduced_features,
)
from glpbench.contracts import SplitBundle

__all__ = [
    "QueryNodeSets",
    "SplitBundle",
    "audit_comparable_splits",
    "calculate_reduced_features",
    "generate_comparable_splits",
    "generate_validation_splits",
    "inductive_split",
    "select_query_node_sets",
    "select_query_nodes",
    "update_reduced_features",
]
