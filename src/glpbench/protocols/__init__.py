"""Transductive and semi-inductive evaluation protocols."""

from glpbench.interfaces import SplitProtocol
from glpbench.protocols.adapters import (
    MatchedSemiInductiveProtocol,
    MatchedValidationProtocol,
    SemiInductiveProtocol,
    TransductiveProtocol,
)
from glpbench.protocols.registry import (
    available_split_protocols,
    create_split_protocol,
    get_split_protocol_factory,
    get_split_protocol_requirements,
    register_split_protocol,
)
from glpbench.protocols.splitting import (
    QueryNodeSets,
    audit_comparable_splits,
    audit_semi_inductive_splits,
    audit_transductive_splits,
    audit_validation_splits,
    calculate_reduced_features,
    generate_comparable_splits,
    generate_transductive_splits,
    generate_validation_splits,
    inductive_split,
    select_query_node_sets,
    select_query_nodes,
    update_reduced_features,
)
from glpbench.contracts import SplitBundle


def _register_builtin_protocols():
    """Register protocol adapters included with GLPBench."""
    for protocol in (
        MatchedSemiInductiveProtocol,
        MatchedValidationProtocol,
        SemiInductiveProtocol,
        TransductiveProtocol,
    ):
        register_split_protocol(protocol.name, protocol, replace=True)


_register_builtin_protocols()


__all__ = [
    "MatchedSemiInductiveProtocol",
    "MatchedValidationProtocol",
    "QueryNodeSets",
    "SemiInductiveProtocol",
    "SplitBundle",
    "SplitProtocol",
    "TransductiveProtocol",
    "available_split_protocols",
    "audit_comparable_splits",
    "audit_semi_inductive_splits",
    "audit_transductive_splits",
    "audit_validation_splits",
    "calculate_reduced_features",
    "create_split_protocol",
    "generate_comparable_splits",
    "generate_transductive_splits",
    "generate_validation_splits",
    "get_split_protocol_factory",
    "get_split_protocol_requirements",
    "inductive_split",
    "register_split_protocol",
    "select_query_node_sets",
    "select_query_nodes",
    "update_reduced_features",
]
