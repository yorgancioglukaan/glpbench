"""Tests for stateless graph-feature normalization."""

import unittest
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import torch
from torch_geometric.data import Data

from glpbench.preprocessing import normalize_graph_features


class TestGraphFeatureNormalization(unittest.TestCase):
    def test_returns_normalized_clone_without_mutating_input(self):
        graph = Data(
            x=torch.tensor([[2.0, 2.0], [1.0, 3.0], [4.0, 0.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        original = graph.x.clone()

        normalized = normalize_graph_features(graph)

        self.assertIsNot(normalized, graph)
        self.assertTrue(torch.equal(graph.x, original))
        self.assertTrue(torch.allclose(
            normalized.x.sum(dim=1), torch.ones(graph.num_nodes)
        ))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(TypeError):
            normalize_graph_features(object())
        with self.assertRaisesRegex(ValueError, "node features"):
            normalize_graph_features(
                Data(edge_index=torch.empty((2, 0), dtype=torch.long))
            )


if __name__ == "__main__":
    unittest.main()
