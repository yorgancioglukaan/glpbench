import unittest
from unittest.mock import patch
import argparse
import copy
import csv
from contextlib import redirect_stdout
from dataclasses import replace
import importlib
from io import StringIO
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import torch
import yaml
import glpbench.protocols.splitting as utils
import glpbench.datasets.builtin as datasets
import glpbench.runners.experiment as main
import glpbench.runners.benchmark as benchmark
import glpbench.runners.audit as convergence_audit
import glpbench.runners.search as search
import glpbench.preprocessing.reduction as feature_reduction
import glpbench.training.scheduling as lr_scheduling
from glpbench import preprocessing, training
import glpbench.datasets as glp_datasets
import glpbench.evaluation as glp_evaluation
import glpbench.logging as glp_logging
import glpbench.models as glp_models
import glpbench.protocols as glp_protocols
import glpbench.runners.experiment as experiment_runner
import glpbench.runners.benchmark as benchmark_runner
import glpbench.runners.search as search_runner
import glpbench.runners.audit as audit_runner
import glpbench.evaluation.pipeline as experiment_pipeline
from glpbench.logging.csv_store import (
    CSVEpochHistoryLogger,
    CSVResultLogger,
    EpochHistoryRecord,
    ExperimentResult,
    make_configuration_id,
)
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, SAGEConv
import glpbench.models.link_prediction as link_prediction


class TestDatasetLoading(unittest.TestCase):
    @staticmethod
    def _graph():
        return Data(
            x=torch.tensor([[2.0, 2.0], [1.0, 3.0], [4.0, 0.0]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )

    def test_available_datasets_has_stable_names(self):
        self.assertEqual(datasets.available_datasets(), [
            "amazon-computers", "amazon-photo", "citeseer", "coauthor-cs",
            "cora", "dblp", "deezer-europe", "pubmed", "wikics"
        ])

    @patch("glpbench.datasets.builtin.Planetoid")
    def test_planetoid_names_dispatch_to_canonical_names(self, mock_planetoid):
        expected_names = {
            "cora": "Cora",
            "CITESEER": "CiteSeer",
            "PubMed": "PubMed",
        }
        for supplied_name, canonical_name in expected_names.items():
            with self.subTest(name=supplied_name):
                mock_planetoid.reset_mock()
                mock_planetoid.return_value.__getitem__.return_value = self._graph()

                data = datasets.load_dataset(
                    supplied_name, root=Path("benchmark-data"), normalize_features=False
                )

                mock_planetoid.assert_called_once_with(
                    root="benchmark-data", name=canonical_name
                )
                self.assertEqual(data.metadata.name, canonical_name)
                self.assertEqual(data.metadata.family, "Planetoid")

    @patch("glpbench.datasets.builtin.Amazon")
    def test_amazon_photo_loader_and_aliases(self, mock_amazon):
        mock_amazon.return_value.__getitem__.return_value = self._graph()
        for supplied_name in ("amazon-photo", "Amazon Photo", "amazon_photo"):
            with self.subTest(name=supplied_name):
                mock_amazon.reset_mock()
                data = datasets.load_dataset(
                    supplied_name, root="benchmark-data", normalize_features=False
                )
                mock_amazon.assert_called_once_with(
                    root="benchmark-data", name="Photo"
                )
                self.assertEqual(data.metadata.name, "AmazonPhoto")
                self.assertEqual(data.metadata.family, "Amazon")

    @patch("glpbench.datasets.builtin.Amazon")
    def test_amazon_computers_loader_and_aliases(self, mock_amazon):
        mock_amazon.return_value.__getitem__.return_value = self._graph()
        for supplied_name in (
            "amazon-computers", "Amazon Computers", "amazon_computers"
        ):
            with self.subTest(name=supplied_name):
                mock_amazon.reset_mock()
                data = datasets.load_dataset(
                    supplied_name, root="benchmark-data", normalize_features=False
                )
                mock_amazon.assert_called_once_with(
                    root="benchmark-data", name="Computers"
                )
                self.assertEqual(data.metadata.name, "AmazonComputers")
                self.assertEqual(data.metadata.family, "Amazon")

    @patch("glpbench.datasets.builtin.Coauthor")
    def test_coauthor_cs_loader_and_aliases(self, mock_coauthor):
        mock_coauthor.return_value.__getitem__.return_value = self._graph()
        for supplied_name in ("coauthor-cs", "Coauthor CS", "coauthor_cs"):
            with self.subTest(name=supplied_name):
                mock_coauthor.reset_mock()
                data = datasets.load_dataset(
                    supplied_name, root="benchmark-data", normalize_features=False
                )
                mock_coauthor.assert_called_once_with(root="benchmark-data", name="CS")
                self.assertEqual(data.metadata.name, "CoauthorCS")
                self.assertEqual(data.metadata.family, "Coauthor")

    @patch("glpbench.datasets.builtin.CitationFull")
    def test_dblp_loader(self, mock_citation_full):
        mock_citation_full.return_value.__getitem__.return_value = self._graph()
        data = datasets.load_dataset(
            "DBLP", root="benchmark-data", normalize_features=False
        )
        mock_citation_full.assert_called_once_with(
            root="benchmark-data", name="DBLP", to_undirected=True
        )
        self.assertEqual(data.metadata.name, "DBLP")
        self.assertEqual(data.metadata.family, "CitationFull")

    @patch("glpbench.datasets.builtin.WikiCS")
    def test_wikics_loader_requests_undirected_graph(self, mock_wikics):
        mock_wikics.return_value.__getitem__.return_value = self._graph()
        data = datasets.load_dataset(
            "wiki-cs", root="benchmark-data", normalize_features=False
        )
        mock_wikics.assert_called_once_with(
            root="benchmark-data", is_undirected=True
        )
        self.assertEqual(data.metadata.name, "WikiCS")
        self.assertEqual(data.metadata.family, "WikiCS")

    @patch("glpbench.datasets.builtin.DeezerEurope")
    def test_deezer_loader_and_alias(self, mock_deezer):
        mock_deezer.return_value.__getitem__.return_value = self._graph()
        data = datasets.load_dataset(
            "Deezer Europe", root="benchmark-data", normalize_features=False
        )
        mock_deezer.assert_called_once_with(root="benchmark-data")
        self.assertEqual(data.metadata.name, "DeezerEurope")
        self.assertEqual(data.metadata.family, "Deezer")

    @patch("glpbench.datasets.builtin.Planetoid")
    def test_loader_clones_normalizes_and_moves_graph(self, mock_planetoid):
        original = self._graph()
        mock_planetoid.return_value.__getitem__.return_value = original

        loaded = datasets.load_dataset("cora", device=torch.device("cpu"))

        self.assertIsNot(loaded.data, original)
        self.assertTrue(torch.allclose(loaded.data.x.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.equal(original.x, self._graph().x))
        self.assertEqual(loaded.data.x.device.type, "cpu")

    @patch("glpbench.datasets.builtin.Planetoid")
    def test_loader_can_skip_feature_normalization(self, mock_planetoid):
        original = self._graph()
        mock_planetoid.return_value.__getitem__.return_value = original

        loaded = datasets.load_dataset("cora", normalize_features=False)

        self.assertTrue(torch.equal(loaded.data.x, original.x))

    @patch("glpbench.datasets.builtin.Planetoid")
    def test_loader_removes_self_loops_without_mutating_source(
        self, mock_planetoid
    ):
        original = self._graph()
        original.edge_index = torch.tensor([
            [0, 0, 1, 1, 1, 2],
            [0, 1, 0, 1, 2, 1],
        ])
        original.edge_attr = torch.arange(6, dtype=torch.float).unsqueeze(1)
        original_edges = original.edge_index.clone()
        original_edge_attr = original.edge_attr.clone()
        mock_planetoid.return_value.__getitem__.return_value = original

        loaded = datasets.load_dataset("cora", normalize_features=False)

        self.assertFalse(torch.any(loaded.data.edge_index[0] == loaded.data.edge_index[1]))
        self.assertTrue(torch.equal(
            loaded.data.edge_index,
            torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ))
        self.assertTrue(torch.equal(
            loaded.data.edge_attr,
            torch.tensor([[1.0], [2.0], [4.0], [5.0]]),
        ))
        self.assertTrue(torch.equal(original.edge_index, original_edges))
        self.assertTrue(torch.equal(original.edge_attr, original_edge_attr))
        self.assertIn("remove-self-loops", loaded.metadata.loader_preprocessing)

    def test_loader_rejects_unknown_name_and_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "Available datasets"):
            datasets.load_dataset("unknown")
        with self.assertRaises(TypeError):
            datasets.load_dataset("cora", root=123)
        with self.assertRaises(TypeError):
            datasets.load_dataset("cora", normalize_features="yes")
        with self.assertRaises(TypeError):
            datasets.load_dataset("cora", device=3)
        with self.assertRaisesRegex(ValueError, "Invalid device"):
            datasets.load_dataset("cora", device="not-a-device")

    @patch("glpbench.datasets.builtin.Planetoid")
    def test_loader_rejects_missing_features(self, mock_planetoid):
        graph = self._graph()
        graph.x = None
        mock_planetoid.return_value.__getitem__.return_value = graph
        with self.assertRaisesRegex(ValueError, "node features"):
            datasets.load_dataset("cora")

    @patch("glpbench.datasets.builtin.Planetoid")
    def test_loader_rejects_directed_graphs(self, mock_planetoid):
        graph = self._graph()
        graph.edge_index = torch.tensor([[0, 1], [1, 2]])
        mock_planetoid.return_value.__getitem__.return_value = graph
        with self.assertRaisesRegex(ValueError, "undirected"):
            datasets.load_dataset("cora")


