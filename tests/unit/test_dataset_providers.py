"""Tests for dataset validation and provider registration."""

import unittest
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from glpbench.contracts import DatasetMetadata, LoadedDataset
from glpbench.datasets import (
    available_datasets,
    get_dataset_provider,
    load_dataset,
    register_dataset_provider,
    validate_graph_data,
)
import glpbench.datasets.registry as registry
from glpbench.exceptions import DatasetError


def _graph():
    return Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        num_nodes=3,
    )


def _loaded(name="ToyGraph", data=None):
    graph = _graph() if data is None else data
    return LoadedDataset(
        graph,
        DatasetMetadata(
            name=name,
            family="test",
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            directed=False,
            has_node_features=graph.x is not None,
            num_node_features=0 if graph.x is None else graph.x.size(1),
        ),
    )


@dataclass
class _Provider:
    name: str
    result: object

    def load(self, root, device):
        return self.result


class _IsolatedRegistryTest(unittest.TestCase):
    def setUp(self):
        self._providers = dict(registry._PROVIDERS)
        self._lookup = dict(registry._NAME_LOOKUP)
        registry._PROVIDERS.clear()
        registry._NAME_LOOKUP.clear()

    def tearDown(self):
        registry._PROVIDERS.clear()
        registry._PROVIDERS.update(self._providers)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)


class TestDatasetProviderRegistry(_IsolatedRegistryTest):
    def test_register_lookup_aliases_and_stable_listing(self):
        beta = _Provider("beta-graph", _loaded("BetaGraph"))
        alpha = _Provider("alpha-graph", _loaded("AlphaGraph"))
        register_dataset_provider(beta, aliases=("Beta Graph", "beta_alias"))
        register_dataset_provider(alpha)

        self.assertIs(get_dataset_provider("BETA-ALIAS"), beta)
        self.assertIs(get_dataset_provider("beta graph"), beta)
        self.assertEqual(available_datasets(), ["alpha-graph", "beta-graph"])

    def test_duplicate_and_alias_collisions_are_rejected_atomically(self):
        first = _Provider("first", _loaded("First"))
        register_dataset_provider(first, aliases=("shared",))

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_dataset_provider(_Provider("first", _loaded("First")))
        with self.assertRaisesRegex(ValueError, "collides"):
            register_dataset_provider(
                _Provider("second", _loaded("Second")), aliases=("shared",)
            )

        self.assertEqual(available_datasets(), ["first"])
        self.assertIs(get_dataset_provider("shared"), first)

    def test_explicit_replacement_removes_stale_aliases(self):
        original = _Provider("toy", _loaded("Toy"))
        replacement = _Provider("toy", _loaded("Toy"))
        register_dataset_provider(original, aliases=("old-alias",))
        register_dataset_provider(
            replacement, aliases=("new-alias",), replace=True
        )

        self.assertIs(get_dataset_provider("new alias"), replacement)
        with self.assertRaisesRegex(DatasetError, "Unknown dataset"):
            get_dataset_provider("old-alias")

    def test_registration_validates_provider_and_options(self):
        with self.assertRaises(TypeError):
            register_dataset_provider(object())
        with self.assertRaises(TypeError):
            register_dataset_provider(_Provider("toy", _loaded()), aliases="alias")
        with self.assertRaises(TypeError):
            register_dataset_provider(_Provider("toy", _loaded()), replace=1)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            register_dataset_provider(_Provider(" ", _loaded()))

    def test_load_dataset_validates_provider_result_and_graph(self):
        register_dataset_provider(_Provider("wrong", object()))
        with self.assertRaisesRegex(DatasetError, "return LoadedDataset"):
            load_dataset("wrong", normalize_features=False)

        directed = _graph()
        directed.edge_index = torch.tensor([[0, 1], [1, 2]])
        register_dataset_provider(
            _Provider("directed", _loaded("Directed", directed))
        )
        with self.assertRaisesRegex(DatasetError, "undirected"):
            load_dataset("directed", normalize_features=False)

    def test_load_dataset_isolates_provider_data(self):
        supplied = _loaded()
        register_dataset_provider(_Provider("toy-graph", supplied))

        loaded = load_dataset("Toy Graph", normalize_features=False)

        self.assertIsNot(loaded.data, supplied.data)
        loaded.data.x[0, 0] = 99
        self.assertEqual(supplied.data.x[0, 0], 1)


class TestGraphDataValidation(unittest.TestCase):
    def test_accepts_valid_graph_without_mutation(self):
        graph = _graph()
        original_x = graph.x.clone()
        original_edges = graph.edge_index.clone()

        self.assertIs(validate_graph_data(graph), graph)
        self.assertTrue(torch.equal(graph.x, original_x))
        self.assertTrue(torch.equal(graph.edge_index, original_edges))

    def test_rejects_invalid_feature_matrices(self):
        cases = {
            "node features": Data(
                edge_index=_graph().edge_index, num_nodes=3
            ),
            "shape": Data(
                x=torch.ones(3), edge_index=_graph().edge_index, num_nodes=3
            ),
            "one row per node": Data(
                x=torch.ones(2, 2), edge_index=_graph().edge_index, num_nodes=3
            ),
            "real numeric": Data(
                x=torch.ones(3, 2, dtype=torch.bool),
                edge_index=_graph().edge_index,
                num_nodes=3,
            ),
            "finite": Data(
                x=torch.tensor([[1.0], [float("nan")], [2.0]]),
                edge_index=_graph().edge_index,
                num_nodes=3,
            ),
        }
        for message, graph in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(DatasetError, message):
                    validate_graph_data(graph)

    def test_rejects_invalid_edge_index(self):
        cases = {
            "shape": torch.tensor([0, 1]),
            "integer dtype": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "outside": torch.tensor([[0, 3], [1, 0]]),
            "self-loops": torch.tensor([[0, 0], [0, 1]]),
            "duplicate": torch.tensor([[0, 1, 0], [1, 0, 1]]),
            "undirected": torch.tensor([[0, 1], [1, 2]]),
        }
        for message, edge_index in cases.items():
            graph = Data(x=torch.ones(3, 2), edge_index=edge_index, num_nodes=3)
            with self.subTest(message=message):
                with self.assertRaisesRegex(DatasetError, message):
                    validate_graph_data(graph)

    def test_optional_constraints_can_be_relaxed(self):
        directed_with_loop = Data(
            edge_index=torch.tensor([[0, 0], [0, 1]]), num_nodes=2
        )
        self.assertIs(
            validate_graph_data(
                directed_with_loop,
                require_features=False,
                require_undirected=False,
                require_simple=False,
            ),
            directed_with_loop,
        )


if __name__ == "__main__":
    unittest.main()
