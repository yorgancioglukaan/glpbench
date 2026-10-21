"""Tests for registered split-protocol adapters."""

import unittest
from dataclasses import dataclass, replace
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import torch
from torch_geometric.data import Data

from glpbench.contracts import AuditReport, SplitBundle
from glpbench.exceptions import ProtocolError
from glpbench.protocols import (
    available_split_protocols,
    create_split_protocol,
    generate_comparable_splits,
    generate_transductive_splits,
    generate_validation_splits,
    get_split_protocol_factory,
    inductive_split,
    register_split_protocol,
)
import glpbench.protocols.registry as registry


def _graph():
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    directed = edges + [(target, source) for source, target in edges]
    return Data(
        x=torch.ones(8, 2),
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        num_nodes=8,
    )


def _assert_partition_equal(test_case, expected, actual):
    for attribute in (
        "edge_index", "edge_label_index", "edge_label",
        "validation_query_mask", "test_query_mask", "visible_node_mask",
    ):
        test_case.assertTrue(torch.equal(
            getattr(expected, attribute), getattr(actual, attribute)
        ))


def _assert_bundle_equal(test_case, expected, actual):
    _assert_partition_equal(test_case, expected.train, actual.train)
    if expected.validation is None:
        test_case.assertIsNone(actual.validation)
    else:
        _assert_partition_equal(test_case, expected.validation, actual.validation)
    test_case.assertEqual(set(expected.tests), set(actual.tests))
    for name in expected.tests:
        _assert_partition_equal(test_case, expected.tests[name], actual.tests[name])
    test_case.assertEqual(
        expected.validation_query_nodes, actual.validation_query_nodes
    )
    test_case.assertEqual(expected.test_query_nodes, actual.test_query_nodes)


@dataclass(frozen=True)
class _Protocol:
    name = "toy"
    value: int = 0

    def split(self, data, seed):
        return data, seed, self.value

    def audit(self, data, splits):
        return AuditReport(True, {"toy": True})


class TestProtocolRegistry(unittest.TestCase):
    def setUp(self):
        self._factories = dict(registry._PROTOCOL_FACTORIES)
        self._lookup = dict(registry._NAME_LOOKUP)

    def tearDown(self):
        registry._PROTOCOL_FACTORIES.clear()
        registry._PROTOCOL_FACTORIES.update(self._factories)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)

    def test_builtin_protocols_have_stable_names(self):
        self.assertEqual(available_split_protocols(), (
            "matched-semi-inductive",
            "matched-semi-inductive-validation-only",
            "semi-inductive",
            "transductive",
        ))

    def test_register_lookup_alias_and_construct(self):
        register_split_protocol("toy", _Protocol, aliases=("Toy Protocol",))

        self.assertIs(get_split_protocol_factory("TOY-PROTOCOL"), _Protocol)
        protocol = create_split_protocol("toy protocol", value=3)
        self.assertEqual(protocol.value, 3)

    def test_duplicate_collision_and_replacement(self):
        register_split_protocol("toy", _Protocol, aliases=("old",))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_split_protocol("toy", _Protocol)
        with self.assertRaisesRegex(ValueError, "collides"):
            register_split_protocol("other", _Protocol, aliases=("old",))

        register_split_protocol(
            "toy", lambda **kwargs: _Protocol(**kwargs),
            aliases=("new",), replace=True,
        )
        self.assertEqual(create_split_protocol("new", value=4).value, 4)
        with self.assertRaises(ProtocolError):
            get_split_protocol_factory("old")

    def test_invalid_factories_and_unknown_names_are_rejected(self):
        with self.assertRaises(TypeError):
            register_split_protocol("invalid", object())
        register_split_protocol("invalid", lambda: object())
        with self.assertRaisesRegex(TypeError, "SplitProtocol"):
            create_split_protocol("invalid")
        with self.assertRaisesRegex(ProtocolError, "Available protocols"):
            create_split_protocol("missing")


class TestProtocolAdapters(unittest.TestCase):
    def test_matched_adapter_matches_procedural_api(self):
        data = _graph()
        parameters = {
            "query": [7],
            "validation_query": [6],
            "train_ratio": 0.75,
        }
        expected = generate_comparable_splits(data, seed=11, **parameters)

        protocol = create_split_protocol(
            "Matched Semi Inductive", **parameters
        )
        actual = protocol.split(data, seed=11)

        _assert_bundle_equal(self, expected, actual)
        self.assertTrue(protocol.audit(data, actual).passed)

    def test_validation_adapter_matches_procedural_api(self):
        data = _graph()
        parameters = {
            "query_size": 1,
            "validation_query_size": 1,
            "train_ratio": 0.75,
        }
        expected = generate_validation_splits(data, seed=12, **parameters)

        protocol = create_split_protocol(
            "matched-semi-inductive-validation-only", **parameters
        )
        actual = protocol.split(data, seed=12)

        _assert_bundle_equal(self, expected, actual)
        self.assertTrue(protocol.audit(data, actual).passed)

    def test_semi_inductive_adapter_matches_procedural_api(self):
        data = _graph()
        parameters = {
            "query": [7],
            "validation_query": [6],
            "train_ratio": 0.75,
        }
        expected = inductive_split(data, seed=13, **parameters)

        protocol = create_split_protocol("semi_inductive", **parameters)
        actual = protocol.split(data, seed=13)

        _assert_bundle_equal(self, expected, actual)
        self.assertTrue(protocol.audit(data, actual).passed)

    def test_validation_audit_rejects_query_leakage(self):
        data = _graph()
        protocol = create_split_protocol(
            "matched-semi-inductive-validation-only",
            query_size=1,
            validation_query_size=1,
            train_ratio=0.75,
        )
        splits = protocol.split(data, seed=4)
        corrupted = splits.validation.clone()
        query = splits.validation_query_nodes[0]
        visible = int(splits.visible_node_mask.nonzero()[0])
        corrupted.edge_index = torch.cat([
            corrupted.edge_index,
            torch.tensor([[visible, query], [query, visible]]),
        ], dim=1)

        with self.assertRaisesRegex(ProtocolError, "query node"):
            protocol.audit(data, replace(splits, validation=corrupted))

    def test_transductive_protocol_is_deterministic_and_audited(self):
        data = _graph()
        protocol = create_split_protocol(
            "transductive", validation_ratio=0.2, test_ratio=0.2
        )

        first = protocol.split(data, seed=14)
        second = generate_transductive_splits(
            data, validation_ratio=0.2, test_ratio=0.2, seed=14
        )

        _assert_bundle_equal(self, first, second)
        self.assertTrue(protocol.audit(data, first).passed)
        self.assertEqual(first.validation_query_nodes, ())
        self.assertEqual(first.test_query_nodes, ())
        for split in (first.validation, first.tests["transductive"]):
            self.assertEqual(
                int((split.edge_label == 1).sum()),
                int((split.edge_label == 0).sum()),
            )

    def test_transductive_protocol_validates_ratios(self):
        data = _graph()
        with self.assertRaisesRegex(ValueError, "sum"):
            generate_transductive_splits(
                data, validation_ratio=0.5, test_ratio=0.5
            )


if __name__ == "__main__":
    unittest.main()
