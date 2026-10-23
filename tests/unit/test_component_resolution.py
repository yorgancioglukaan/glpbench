"""Tests for registry-backed evaluation component resolution."""

import unittest
from dataclasses import dataclass
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch_geometric.data import Data


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.contracts import AuditReport
from glpbench.evaluation import realize_registered_split
from glpbench.exceptions import ProtocolError
from glpbench.protocols import generate_validation_splits, register_split_protocol
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


@dataclass
class _Protocol:
    name = "test-protocol"
    report: object
    calls: list

    def split(self, data, seed, *, dataset_fingerprint=None):
        self.calls.append(("split", seed, dataset_fingerprint))
        return generate_validation_splits(
            data,
            query_size=1,
            validation_query_size=1,
            train_ratio=0.75,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
        )

    def audit(self, data, splits):
        self.calls.append(("audit", splits.metadata.fingerprint))
        return self.report


class TestComponentResolution(unittest.TestCase):
    def setUp(self):
        self._factories = dict(registry._PROTOCOL_FACTORIES)
        self._lookup = dict(registry._NAME_LOOKUP)

    def tearDown(self):
        registry._PROTOCOL_FACTORIES.clear()
        registry._PROTOCOL_FACTORIES.update(self._factories)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)

    def test_registered_protocol_is_constructed_split_and_audited(self):
        calls = []
        register_split_protocol(
            "test-protocol",
            lambda: _Protocol(AuditReport(True, {"valid": True}), calls),
        )

        splits = realize_registered_split(
            "test-protocol",
            _graph(),
            seed=17,
            dataset_fingerprint="sha256:dataset",
        )

        self.assertEqual(calls[0], ("split", 17, "sha256:dataset"))
        self.assertEqual(calls[1][0], "audit")
        self.assertEqual(splits.tests, {})

    def test_failed_or_malformed_audit_stops_resolution(self):
        for report, message in (
            (AuditReport(False, {"valid": False}), "valid"),
            (True, "AuditReport"),
        ):
            with self.subTest(report=report):
                register_split_protocol(
                    "test-protocol",
                    lambda report=report: _Protocol(report, []),
                    replace=True,
                )
                with self.assertRaisesRegex(ProtocolError, message):
                    realize_registered_split(
                        "test-protocol", _graph(), seed=17
                    )

    def test_builtin_resolution_matches_procedural_validation_split(self):
        data = _graph()
        parameters = {
            "query_size": 1,
            "validation_query_size": 1,
            "train_ratio": 0.75,
        }
        expected = generate_validation_splits(data, seed=19, **parameters)

        with patch(
            "glpbench.evaluation.components.create_split_protocol",
            wraps=__import__(
                "glpbench.protocols", fromlist=["create_split_protocol"]
            ).create_split_protocol,
        ) as create:
            actual = realize_registered_split(
                "matched-semi-inductive-validation-only",
                data,
                seed=19,
                parameters=parameters,
            )

        create.assert_called_once_with(
            "matched-semi-inductive-validation-only", **parameters
        )
        for expected_split, actual_split in (
            (expected.train, actual.train),
            (expected.validation, actual.validation),
        ):
            for attribute in ("edge_index", "edge_label_index", "edge_label"):
                self.assertTrue(torch.equal(
                    getattr(expected_split, attribute),
                    getattr(actual_split, attribute),
                ))


if __name__ == "__main__":
    unittest.main()
