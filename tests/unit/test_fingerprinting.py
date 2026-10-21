"""Tests for deterministic dataset and split fingerprints."""

import unittest
from dataclasses import replace

import torch
from torch_geometric.data import Data

from glpbench.contracts import DatasetMetadata
from glpbench.exceptions import ProtocolError
from glpbench.fingerprinting import fingerprint_dataset, fingerprint_split
from glpbench.protocols import create_split_protocol, generate_comparable_splits


def _graph():
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    directed = edges + [(target, source) for source, target in edges]
    return Data(
        x=torch.arange(16, dtype=torch.float32).reshape(8, 2),
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        num_nodes=8,
    )


def _metadata(version="1"):
    return DatasetMetadata(
        name="toy",
        family="synthetic",
        num_nodes=8,
        num_edges=24,
        directed=False,
        has_node_features=True,
        num_node_features=2,
        dataset_version=version,
    )


class TestDatasetFingerprinting(unittest.TestCase):
    def test_is_deterministic_and_edge_order_independent(self):
        data = _graph()
        reordered = data.clone()
        reordered.edge_index = reordered.edge_index[:, torch.arange(
            reordered.edge_index.size(1) - 1, -1, -1
        )]

        first = fingerprint_dataset(data, _metadata())
        self.assertEqual(first, fingerprint_dataset(data.clone(), _metadata()))
        self.assertEqual(first, fingerprint_dataset(reordered, _metadata()))
        self.assertTrue(first.startswith("sha256:"))

    def test_changes_with_features_edges_and_version(self):
        data = _graph()
        baseline = fingerprint_dataset(data, _metadata())

        changed_features = data.clone()
        changed_features.x[0, 0] += 1
        changed_edges = data.clone()
        changed_edges.edge_index[:, :2] = torch.tensor([[0, 2], [2, 0]])

        self.assertNotEqual(
            baseline, fingerprint_dataset(changed_features, _metadata())
        )
        self.assertNotEqual(
            baseline, fingerprint_dataset(changed_edges, _metadata())
        )
        self.assertNotEqual(
            baseline, fingerprint_dataset(data, _metadata(version="2"))
        )


class TestSplitFingerprinting(unittest.TestCase):
    def _splits(self, seed=7):
        data = _graph()
        dataset_fingerprint = fingerprint_dataset(data, _metadata())
        return data, generate_comparable_splits(
            data,
            query=[7],
            validation_query=[6],
            train_ratio=0.75,
            seed=seed,
            dataset_fingerprint=dataset_fingerprint,
        )

    def test_constructor_records_deterministic_fingerprint(self):
        _, first = self._splits()
        _, second = self._splits()

        self.assertEqual(first.metadata.fingerprint, second.metadata.fingerprint)
        self.assertEqual(
            first.metadata.fingerprint,
            fingerprint_split(
                first, first.metadata.parameters["dataset_fingerprint"]
            ),
        )

    def test_candidate_column_order_does_not_change_fingerprint(self):
        _, splits = self._splits()
        test = splits.tests["semi-inductive"].clone()
        order = torch.arange(test.edge_label.numel() - 1, -1, -1)
        test.edge_label_index = test.edge_label_index[:, order]
        test.edge_label = test.edge_label[order]
        reordered = replace(
            splits,
            tests={**splits.tests, "semi-inductive": test},
        )

        dataset_fingerprint = splits.metadata.parameters["dataset_fingerprint"]
        self.assertEqual(
            fingerprint_split(splits, dataset_fingerprint),
            fingerprint_split(reordered, dataset_fingerprint),
        )

    def test_changes_with_seed_query_and_candidate_contents(self):
        _, baseline = self._splits(seed=7)
        _, changed_seed = self._splits(seed=8)
        data = _graph()
        dataset_fingerprint = fingerprint_dataset(data, _metadata())
        changed_query = generate_comparable_splits(
            data,
            query=[6],
            validation_query=[7],
            train_ratio=0.75,
            seed=7,
            dataset_fingerprint=dataset_fingerprint,
        )
        changed_candidate = baseline.tests["semi-inductive"].clone()
        changed_candidate.edge_label_index[0, 0] = 1
        changed_bundle = replace(
            baseline,
            tests={**baseline.tests, "semi-inductive": changed_candidate},
        )

        self.assertNotEqual(
            baseline.metadata.fingerprint, changed_seed.metadata.fingerprint
        )
        self.assertNotEqual(
            baseline.metadata.fingerprint, changed_query.metadata.fingerprint
        )
        self.assertNotEqual(
            baseline.metadata.fingerprint,
            fingerprint_split(changed_bundle, dataset_fingerprint),
        )

    def test_audit_rejects_tampered_fingerprint(self):
        data, splits = self._splits()
        corrupted = replace(
            splits,
            metadata=replace(splits.metadata, fingerprint="sha256:invalid"),
        )
        protocol = create_split_protocol(
            "matched-semi-inductive",
            query=[7],
            validation_query=[6],
            train_ratio=0.75,
        )

        with self.assertRaisesRegex(ProtocolError, "fingerprint"):
            protocol.audit(data, corrupted)


if __name__ == "__main__":
    unittest.main()
