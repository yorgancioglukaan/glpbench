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
from glpbench.contracts import DatasetMetadata, LoadedDataset
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


def _loaded_dataset(graph):
    return LoadedDataset(
        graph,
        DatasetMetadata(
            name="Toy",
            family="Test",
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            directed=False,
            has_node_features=graph.x is not None,
            num_node_features=graph.num_features,
        ),
    )


class TestSearchRunner(unittest.TestCase):
    @staticmethod
    def _graph(num_nodes=20):
        edges = [(index, (index + 1) % num_nodes) for index in range(num_nodes)]
        edges += [(index, (index + 4) % num_nodes) for index in range(num_nodes)]
        directed = edges + [(target, source) for source, target in edges]
        edge_index = torch.unique(
            torch.tensor(directed, dtype=torch.long).t(), dim=1
        )
        data = Data(
            x=torch.arange(num_nodes * 4, dtype=torch.float).reshape(num_nodes, 4) + 1,
            edge_index=edge_index,
        )
        data.dataset_name = "Toy"
        data.dataset_family = "Test"
        return data

    def test_pilot_spec_generates_deterministic_stratified_candidates(self):
        config = search.load_search_config("tests/fixtures/pilot_search.yaml")
        self.assertEqual(config.raw["training"]["scheduler"], {"type": "none"})
        first = search.generate_candidates(config)
        second = search.generate_candidates(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8 * 7 * 3)
        for dataset in config.datasets:
            for architecture in config.architectures:
                group = [
                    candidate for candidate in first
                    if candidate.dataset == dataset
                    and candidate.architecture == architecture
                ]
                self.assertEqual(len(group), 3)
                self.assertEqual(
                    {candidate.hyperparameters["normalize_features"] for candidate in group},
                    {True, False},
                )
                self.assertEqual(group[0].hyperparameters["hidden_channels"], 128)
                self.assertEqual(group[0].hyperparameters["learning_rate"], 0.01)
                if architecture in {"asymmetric", "asymmetric-mlp"}:
                    self.assertEqual(
                        {candidate.hyperparameters["reduction_method"] for candidate in group},
                        {"pca", "truncated-svd", "sparse-random-projection"},
                    )
                    self.assertEqual(
                        {candidate.hyperparameters["regularization_strength"] for candidate in group},
                        {0.0, 0.001, 0.01},
                    )
                if architecture == "variational":
                    self.assertEqual(
                        {candidate.hyperparameters["kl_weight"] for candidate in group},
                        {0.01, 0.1, 1.0},
                    )
                if architecture == "deal":
                    self.assertEqual(
                        {candidate.hyperparameters["deal_profile"] for candidate in group},
                        {"balanced", "attribute-focused", "alignment-focused"},
                    )
                if architecture in {"mlp", "feature-mlp", "asymmetric-mlp"}:
                    self.assertIn(
                        group[0].hyperparameters["decoder_hidden_channels"],
                        {32, 64, 128},
                    )
                    self.assertIn(
                        group[0].hyperparameters["decoder_dropout"],
                        {0.0, 0.2, 0.5},
                    )

    def test_search_accepts_a_fixed_scheduler_configuration(self):
        document = yaml.safe_load(
            Path("tests/fixtures/pilot_search.yaml").read_text(encoding="utf-8")
        )
        document["training"]["scheduler"] = {
            "type": "cosine-annealing",
            "minimum_learning_rate": 1e-5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            config = search.load_search_config(path)

        self.assertEqual(config.raw["training"]["scheduler"], {
            "type": "cosine-annealing",
            "minimum_learning_rate": 1e-5,
        })

    def test_validation_data_is_isolated_and_does_not_expose_test_splits(self):
        source = self._graph()
        original_features = source.x.clone()
        splits = utils.generate_comparable_splits(
            source,
            query_fraction=0.05,
            validation_query_fraction=0.05,
            seed=4,
        )
        with patch(
            "glpbench.preprocessing.adapters.fit_reducer",
            wraps=utils.fit_reducer,
        ) as reducer:
            prepared = experiment_pipeline.prepare_validation_data(
                source,
                splits,
                model_type="asymmetric",
                normalize_features=True,
                out_channels=2,
                reduction_method="pca",
                random_state=4,
            )

        self.assertEqual(prepared.splits.tests, {})
        self.assertTrue(torch.equal(source.x, original_features))
        self.assertTrue(torch.equal(
            prepared.splits.train.edge_index, splits.train.edge_index
        ))
        self.assertTrue(torch.equal(
            reducer.call_args.kwargs["fit_nodes"], splits.train.visible_node_mask
        ))
        self.assertEqual(
            prepared.splits.train.x_reduced.shape, (source.num_nodes, 2)
        )
        self.assertIsNotNone(prepared.preprocessing_time_seconds)

    def test_validation_only_foundation_bypasses_full_candidate_preparation(self):
        source = self._graph()
        splits = utils.generate_validation_splits(
            source,
            query_fraction=0.05,
            validation_query_fraction=0.05,
            seed=4,
        )
        with patch(
            "glpbench.evaluation.pipeline.prepare_candidate_data"
        ) as full_preparation:
            prepared = experiment_pipeline.prepare_validation_data(
                source,
                splits,
                model_type="symmetric",
                normalize_features=True,
                out_channels=2,
                random_state=4,
            )

        full_preparation.assert_not_called()
        self.assertEqual(prepared.splits.tests, {})
        self.assertTrue(torch.equal(
            prepared.splits.train.edge_index, splits.train.edge_index
        ))

    @patch("glpbench.runners.search.load_registered_dataset")
    def test_search_resolves_validation_only_protocol(self, mock_load):
        mock_load.return_value = _loaded_dataset(self._graph())
        config = search.load_search_config("tests/fixtures/pilot_search.yaml")
        source_cache = {}
        split_cache = {}

        first = search._prepare_search_split(
            config, "cora", 4, source_cache, split_cache
        )
        second = search._prepare_search_split(
            config, "cora", 4, source_cache, split_cache
        )

        self.assertIs(first, second)
        self.assertEqual(
            first.metadata.name,
            "matched-semi-inductive-validation-only",
        )
        self.assertEqual(first.tests, {})
        mock_load.assert_called_once()

    def test_selection_uses_tolerance_then_ap_then_parameter_count(self):
        base = search.load_search_config("tests/fixtures/pilot_search.yaml")
        candidates = [
            candidate for candidate in search.generate_candidates(base)
            if candidate.dataset == "cora" and candidate.architecture == "symmetric"
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base,
                datasets=("cora",),
                architectures=("symmetric",),
                history_path=Path(directory) / "history.csv",
                failures_path=Path(directory) / "errors.jsonl",
                aggregate_path=Path(directory) / "aggregates.csv",
                selected_path=Path(directory) / "selected.yaml",
            )
            scores = ((0.9000, 0.70, 100), (0.8995, 0.80, 200), (0.80, 0.90, 50))
            rows = []
            for candidate, (roc_auc, average_precision, parameters) in zip(
                candidates, scores
            ):
                for seed in config.tuning_seeds:
                    rows.append({
                        "candidate_id": candidate.candidate_id,
                        "tuning_seed": str(seed),
                        "status": "success",
                        "validation_roc_auc": str(roc_auc),
                        "validation_average_precision": str(average_precision),
                        "trainable_parameter_count": str(parameters),
                    })

            winners = search.aggregate_and_select(
                config, candidates, rows, "search-1"
            )
            selected = yaml.safe_load(
                config.selected_path.read_text(encoding="utf-8")
            )

        winner = winners[("cora", "symmetric")]
        self.assertEqual(winner["candidate_id"], candidates[1].candidate_id)
        self.assertEqual(
            selected["selections"]["cora"]["symmetric"]["candidate_id"],
            candidates[1].candidate_id,
        )

    @patch("glpbench.runners.search.load_registered_dataset")
    def test_toy_search_logs_selects_and_resumes(self, mock_load):
        mock_load.return_value = _loaded_dataset(self._graph())
        base = search.load_search_config("tests/fixtures/pilot_search.yaml")
        raw = copy.deepcopy(base.raw)
        raw["training"]["max_epochs"] = 1
        raw["training"]["early_stopping"]["patience"] = 1
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base,
                raw=raw,
                datasets=("cora",),
                architectures=("symmetric",),
                tuning_seeds=(1,),
                history_path=Path(directory) / "history.csv",
                failures_path=Path(directory) / "errors.jsonl",
                aggregate_path=Path(directory) / "aggregates.csv",
                selected_path=Path(directory) / "selected.yaml",
            )
            candidate = search.SearchCandidate(
                "toy-candidate", "cora", "symmetric",
                {
                    "hidden_channels": 4,
                    "out_channels": 2,
                    "learning_rate": 0.01,
                    "weight_decay": 0.001,
                    "normalize_features": False,
                    "conv_layer_type": "GCNConv",
                },
            )
            first = search.execute_search(
                config, (candidate,), run_id="pilot-1"
            )
            second = search.execute_search(
                config, (candidate,), resume=True
            )
            with config.history_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(first.succeeded, 1)
        self.assertEqual(first.selected, 1)
        self.assertEqual(second.skipped, 1)
        self.assertEqual(second.run_id, "pilot-1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], "toy-candidate")

    @patch("glpbench.runners.search.realize_registered_split")
    @patch("glpbench.runners.search.load_registered_dataset")
    def test_shared_split_failure_is_cached_and_logged_once(
        self, mock_load, mock_split
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        mock_split.side_effect = ValueError("infeasible shared split")
        base = search.load_search_config("tests/fixtures/pilot_search.yaml")
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base,
                datasets=("cora",),
                architectures=("symmetric",),
                tuning_seeds=(1,),
                history_path=Path(directory) / "history.csv",
                failures_path=Path(directory) / "errors.jsonl",
                aggregate_path=Path(directory) / "aggregates.csv",
                selected_path=Path(directory) / "selected.yaml",
            )
            candidates = (
                search.SearchCandidate(
                    "candidate-1", "cora", "symmetric",
                    {
                        "hidden_channels": 4,
                        "out_channels": 2,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "normalize_features": False,
                        "conv_layer_type": "GCNConv",
                    },
                ),
                search.SearchCandidate(
                    "candidate-2", "cora", "symmetric",
                    {
                        "hidden_channels": 8,
                        "out_channels": 2,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "normalize_features": False,
                        "conv_layer_type": "GCNConv",
                    },
                ),
            )

            summary = search.execute_search(
                config, candidates, run_id="shared-failure"
            )
            records = [
                json.loads(line)
                for line in config.failures_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary.total_runs, 2)
        self.assertEqual(summary.failed, 2)
        self.assertEqual(summary.succeeded, 0)
        self.assertEqual(mock_split.call_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scope"], "dataset-seed")
        self.assertEqual(records[0]["stage"], "data-splitting")
        self.assertEqual(records[0]["dataset"], "cora")
        self.assertEqual(records[0]["tuning_seed"], 1)
        self.assertEqual(records[0]["affected_runs"], 2)


