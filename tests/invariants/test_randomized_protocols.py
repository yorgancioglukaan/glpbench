"""Randomized invariant checks for built-in split protocols."""

import random
import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

from glpbench.protocols import create_split_protocol


def _random_graph(graph_seed, *, num_nodes=14):
    """Return a deterministic connected simple undirected graph."""
    rng = random.Random(graph_seed)
    pairs = {
        (node, (node + 1) % num_nodes) for node in range(num_nodes)
    }
    pairs = {tuple(sorted(pair)) for pair in pairs}
    candidates = [
        (source, target)
        for source in range(num_nodes)
        for target in range(source + 1, num_nodes)
        if (source, target) not in pairs
    ]
    pairs.update(rng.sample(candidates, num_nodes // 2))
    directed = [
        edge
        for source, target in sorted(pairs)
        for edge in ((source, target), (target, source))
    ]
    return Data(
        x=torch.arange(num_nodes * 3, dtype=torch.float32).reshape(num_nodes, 3),
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        num_nodes=num_nodes,
    )


def _pairs(edge_index):
    return {
        (min(source, target), max(source, target))
        for source, target in edge_index.t().tolist()
    }


def _partition_items(bundle):
    items = [("train", bundle.train)]
    if bundle.validation is not None:
        items.append(("validation", bundle.validation))
    items.extend(bundle.tests.items())
    return items


def _assert_bundles_equal(test_case, left, right):
    test_case.assertEqual(left.validation_query_nodes, right.validation_query_nodes)
    test_case.assertEqual(left.test_query_nodes, right.test_query_nodes)
    test_case.assertEqual(set(left.tests), set(right.tests))
    test_case.assertEqual(left.metadata.fingerprint, right.metadata.fingerprint)
    for (left_name, left_split), (right_name, right_split) in zip(
        _partition_items(left), _partition_items(right), strict=True
    ):
        test_case.assertEqual(left_name, right_name)
        for attribute in (
            "edge_index", "edge_label_index", "edge_label",
            "validation_query_mask", "test_query_mask", "visible_node_mask",
        ):
            test_case.assertTrue(torch.equal(
                getattr(left_split, attribute), getattr(right_split, attribute)
            ))


class TestRandomizedProtocolInvariants(unittest.TestCase):
    """Exercise protocol contracts over varied deterministic toy graphs."""

    PROTOCOLS = {
        "matched-semi-inductive": {
            "query_size": 1,
            "validation_query_size": 1,
            "train_ratio": 0.75,
        },
        "matched-semi-inductive-validation-only": {
            "query_size": 1,
            "validation_query_size": 1,
            "train_ratio": 0.75,
        },
        "semi-inductive": {
            "query_size": 1,
            "validation_query_size": 1,
            "train_ratio": 0.75,
        },
        "transductive": {
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
        },
    }

    def _assert_independent_invariants(self, data, bundle):
        original = _pairs(data.edge_index)
        visible = set(bundle.visible_node_mask.nonzero().view(-1).tolist())
        validation_queries = set(bundle.validation_query_nodes)
        test_queries = set(bundle.test_query_nodes)
        positive_sets = {}

        self.assertTrue(validation_queries.isdisjoint(test_queries))
        self.assertEqual(
            visible | validation_queries | test_queries,
            set(range(data.num_nodes)),
        )

        for name, split in _partition_items(bundle):
            self.assertTrue(is_undirected(split.edge_index))
            self.assertTrue(set(split.edge_index.flatten().tolist()) <= visible)
            self.assertTrue(_pairs(split.edge_index) <= original)
            self.assertEqual(split.edge_label_index.size(1), split.edge_label.numel())
            self.assertTrue(set(split.edge_label.tolist()) <= {0.0, 1.0})

            positives = split.edge_label_index[:, split.edge_label == 1]
            negatives = split.edge_label_index[:, split.edge_label == 0]
            positive_sets[name] = _pairs(positives)
            negative_pairs = _pairs(negatives)
            self.assertEqual(len(positive_sets[name]), positives.size(1))
            self.assertEqual(len(negative_pairs), negatives.size(1))
            self.assertTrue(positive_sets[name] <= original)
            self.assertTrue(negative_pairs.isdisjoint(original))
            self.assertTrue(all(source != target for source, target in negative_pairs))

            if name != "train":
                self.assertTrue(positive_sets[name].isdisjoint(_pairs(split.edge_index)))
                self.assertEqual(positives.size(1), negatives.size(1))

            target_queries = None
            if name == "validation" and validation_queries:
                target_queries = validation_queries
            elif name == "semi-inductive" and test_queries:
                target_queries = test_queries
            if target_queries is not None and split.edge_label_index.numel():
                sources, targets = split.edge_label_index.tolist()
                self.assertTrue(set(sources) <= visible)
                self.assertTrue(set(targets) <= target_queries)

        names = list(positive_sets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                self.assertTrue(positive_sets[left].isdisjoint(positive_sets[right]))

        if "transductive" in bundle.tests:
            endpoints = set(bundle.tests["transductive"].edge_label_index.flatten().tolist())
            self.assertTrue(endpoints <= visible)

    def test_randomized_graphs_satisfy_independent_protocol_invariants(self):
        for graph_seed in range(12):
            data = _random_graph(graph_seed)
            for protocol_name, parameters in self.PROTOCOLS.items():
                with self.subTest(graph_seed=graph_seed, protocol=protocol_name):
                    protocol = create_split_protocol(protocol_name, **parameters)
                    bundle = protocol.split(data, seed=100 + graph_seed)

                    self.assertTrue(protocol.audit(data, bundle).passed)
                    self._assert_independent_invariants(data, bundle)

    def test_randomized_protocol_realizations_are_deterministic(self):
        for graph_seed in range(4):
            data = _random_graph(graph_seed)
            for protocol_name, parameters in self.PROTOCOLS.items():
                with self.subTest(graph_seed=graph_seed, protocol=protocol_name):
                    protocol = create_split_protocol(protocol_name, **parameters)
                    first = protocol.split(data, seed=200 + graph_seed)
                    second = protocol.split(data.clone(), seed=200 + graph_seed)
                    _assert_bundles_equal(self, first, second)


if __name__ == "__main__":
    unittest.main()
