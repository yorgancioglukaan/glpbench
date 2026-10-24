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
import glpbench.models.registry as model_registry


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


class TestMainPipeline(unittest.TestCase):
    @staticmethod
    def _graph():
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        directed_edges = edges + [(target, source) for source, target in edges]
        data = Data(
            x=torch.rand(8, 4),
            edge_index=torch.tensor(directed_edges).t().contiguous(),
        )
        data.dataset_name = "Toy"
        data.dataset_family = "Test"
        return data

    def test_seed_resolution_supports_repeated_and_generated_seeds(self):
        self.assertEqual(
            main._resolve_seeds(SimpleNamespace(seed=[3, 7], num_seeds=None)),
            (3, 7),
        )
        self.assertEqual(
            main._resolve_seeds(SimpleNamespace(seed=None, num_seeds=3)),
            (0, 1, 2),
        )
        self.assertEqual(
            main._resolve_seeds(SimpleNamespace(seed=None, num_seeds=None)),
            (42,),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            main._resolve_seeds(SimpleNamespace(seed=[3, 3], num_seeds=None))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            main._resolve_seeds(SimpleNamespace(seed=[3], num_seeds=2))

    def test_gatv2_cli_options_require_a_json_object(self):
        self.assertEqual(
            main._json_object('{"heads": 2, "dropout": 0.1}'),
            {"heads": 2, "dropout": 0.1},
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            main._json_object("not-json")
        with self.assertRaises(argparse.ArgumentTypeError):
            main._json_object("[1, 2]")

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_local_model_file_runs_and_logs_source_provenance(self, mock_load):
        mock_load.return_value = _loaded_dataset(self._graph())
        adapters = dict(model_registry._MODEL_ADAPTERS)
        lookup = dict(model_registry._NAME_LOOKUP)
        sources = dict(model_registry._MODEL_SOURCES)
        loaded_files = dict(model_registry._LOADED_MODEL_FILES)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model_path = root / "local_cosine.py"
                model_path.write_text(
                    "from glpbench.models import FeatureCosineAdapter\n"
                    "class LocalAdapter(FeatureCosineAdapter):\n"
                    "    name = 'local-cosine'\n"
                    "    def result_metadata(self, parameters):\n"
                    "        self.validate_parameters(parameters)\n"
                    "        return {'custom_marker': 'from-adapter'}\n"
                    "MODEL_ADAPTER = LocalAdapter()\n",
                    encoding="utf-8",
                )
                output = root / "results.csv"
                args = SimpleNamespace(
                    dataset="toy",
                    out_filepath=str(output),
                    data_root="unused",
                    model_type=[str(model_path)],
                    conv_layer_type="GCNConv",
                    hidden_channels=4,
                    out_channels=2,
                    query_size=1,
                    train_ratio=0.8,
                    epochs=1,
                    seed=[3],
                    num_seeds=None,
                    device="cpu",
                    normalize_features=False,
                )

                main.run_experiment(args)
                with output.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))

            self.assertEqual({row["model_type"] for row in rows}, {
                "local-cosine"
            })
            extra = json.loads(rows[0]["extra_fields"])
            self.assertEqual(extra["custom_marker"], "from-adapter")
            self.assertEqual(
                extra["model_source"]["path"], str(model_path.resolve())
            )
            self.assertEqual(len(extra["model_source"]["sha256"]), 64)
        finally:
            model_registry._MODEL_ADAPTERS.clear()
            model_registry._MODEL_ADAPTERS.update(adapters)
            model_registry._NAME_LOOKUP.clear()
            model_registry._NAME_LOOKUP.update(lookup)
            model_registry._MODEL_SOURCES.clear()
            model_registry._MODEL_SOURCES.update(sources)
            model_registry._LOADED_MODEL_FILES.clear()
            model_registry._LOADED_MODEL_FILES.update(loaded_files)

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_multi_seed_run_preserves_pairing_and_configuration_ids(
        self, mock_load
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric", "feature-cosine"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=[1, 2],
                num_seeds=None,
                device="cpu",
                normalize_features=False,
            )
            with patch(
                "glpbench.runners.experiment.realize_registered_split",
                wraps=main.realize_registered_split,
            ) as mock_splits, patch(
                "glpbench.runners.experiment.validate_registered_compatibility",
                wraps=main.validate_registered_compatibility,
            ) as mock_compatibility:
                run_id = main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_splits.call_count, 2)
        self.assertEqual(mock_compatibility.call_count, 2)
        self.assertEqual(
            {
                (call.args[0], call.args[1])
                for call in mock_compatibility.call_args_list
            },
            {
                ("symmetric", "matched-semi-inductive"),
                ("feature-cosine", "matched-semi-inductive"),
            },
        )
        self.assertEqual(
            [call.kwargs["seed"] for call in mock_splits.call_args_list],
            [1, 2],
        )
        self.assertTrue(all(
            call.args[0] == "matched-semi-inductive"
            for call in mock_splits.call_args_list
        ))
        self.assertTrue(all(
            call.kwargs["parameters"]["query_size"] == 1
            for call in mock_splits.call_args_list
        ))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["run_id"] for row in rows}, {run_id})
        self.assertEqual({row["random_seed"] for row in rows}, {"1", "2"})
        self.assertEqual({row["normalize_features"] for row in rows}, {"False"})

        grouped = {}
        for row in rows:
            key = (row["model_type"], row["split_type"])
            grouped.setdefault(key, []).append(row)
        self.assertEqual(len(grouped), 4)
        for repetitions in grouped.values():
            self.assertEqual(len(repetitions), 2)
            self.assertEqual(
                len({row["configuration_id"] for row in repetitions}), 1
            )
            self.assertEqual(
                {row["random_seed"] for row in repetitions}, {"1", "2"}
            )

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_completed_results_skip_model_evaluation_and_duplicate_rows(
        self, mock_load
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["feature-cosine"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=5,
                device="cpu",
                normalize_features=False,
            )
            main.run_experiment(args, run_id="first")
            with output.open(newline="", encoding="utf-8") as csv_file:
                original_rows = list(csv.DictReader(csv_file))
            completed = {
                (row["configuration_id"], int(row["random_seed"]))
                for row in original_rows
            }

            with (
                patch("glpbench.runners.experiment.train_with_validation_selection") as mock_train,
                patch("glpbench.runners.experiment.calculate_metrics") as mock_metrics,
            ):
                main.run_experiment(
                    args,
                    run_id="resume",
                    completed_results=completed,
                )
            with output.open(newline="", encoding="utf-8") as csv_file:
                resumed_rows = list(csv.DictReader(csv_file))

        mock_train.assert_not_called()
        mock_metrics.assert_not_called()
        self.assertEqual(resumed_rows, original_rows)

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_checkpoint_selection_restores_best_epoch_before_testing(
        self, mock_load
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        evaluation_calls = []
        epoch_records = []

        class FakePredictor:
            def __init__(self, train_data, **kwargs):
                self.trainable = True
                self.model_spec = SimpleNamespace(uses_conv_layer=True)
                self.model_parameters = kwargs["model_parameters"]
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                epoch = int(self.model.weight.item())
                evaluation_calls.append((data.split_stage, epoch))
                correct = torch.where(
                    data.edge_label == 1,
                    torch.tensor(0.9, device=data.edge_label.device),
                    torch.tensor(0.1, device=data.edge_label.device),
                )
                if data.split_stage == "validation" and epoch != 2:
                    return 1.0 - correct
                return correct

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=3,
                selection_metric="roc_auc",
                training_policy="current",
                epoch_callback=epoch_records.append,
                seed=42,
                device="cpu",
                normalize_features=False,
            )
            with patch("glpbench.runners.experiment.LinkPrediction", FakePredictor):
                main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(evaluation_calls, [
            ("validation", 1),
            ("validation", 2),
            ("validation", 3),
            ("test", 2),
            ("test", 2),
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["epochs_trained"] for row in rows}, {"3"})
        self.assertEqual({row["best_epoch"] for row in rows}, {"2"})
        self.assertEqual({row["selection_metric"] for row in rows}, {"roc_auc"})
        self.assertEqual(
            {float(row["best_validation_score"]) for row in rows}, {1.0}
        )
        self.assertEqual(
            {row["early_stopping_patience"] for row in rows}, {""}
        )
        self.assertEqual(
            {float(row["early_stopping_min_delta"]) for row in rows}, {0.0}
        )
        self.assertEqual({row["stopped_early"] for row in rows}, {"False"})
        self.assertEqual([record["epoch"] for record in epoch_records], [1, 2, 3])
        self.assertEqual(
            {record["dataset"] for record in epoch_records}, {"Toy"}
        )
        self.assertEqual(
            {record["model_type"] for record in epoch_records}, {"symmetric"}
        )
        self.assertEqual(
            {record["random_seed"] for record in epoch_records}, {42}
        )
        self.assertEqual(
            {record["training_policy"] for record in epoch_records}, {"current"}
        )

    def test_early_stopping_honors_patience_and_restores_best_state(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        train_data = Data(edge_index=torch.empty((2, 0), dtype=torch.long))
        validation_data = Data(edge_label=torch.tensor([1.0, 0.0]))
        validation_scores = [0.50, 0.70, 0.65, 0.60, 0.90]

        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ) as mock_negatives,
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[
                    {"roc_auc": score} for score in validation_scores
                ],
            ) as mock_metrics,
        ):
            summary = main.train_with_validation_selection(
                predictor,
                train_data,
                validation_data,
                training_policy=main.TrainingPolicy(
                    max_epochs=5,
                    selection_metric="roc_auc",
                    early_stopping_patience=2,
                ),
                seed=4,
                device="cpu",
            )

        self.assertEqual(summary.epochs_executed, 4)
        self.assertEqual(summary.best_epoch, 2)
        self.assertEqual(summary.best_validation_score, 0.70)
        self.assertTrue(summary.stopped_early)
        self.assertEqual(predictor.model.weight.item(), 2.0)
        self.assertEqual(mock_negatives.call_count, 4)
        self.assertEqual(mock_metrics.call_count, 4)

    def test_early_stopping_min_delta_ignores_small_improvements(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ),
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[{"roc_auc": 0.50}, {"roc_auc": 0.55}],
            ),
        ):
            summary = main.train_with_validation_selection(
                predictor,
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                training_policy=main.TrainingPolicy(
                    max_epochs=5,
                    selection_metric="roc_auc",
                    early_stopping_patience=1,
                    early_stopping_min_delta=0.10,
                ),
                seed=4,
                device="cpu",
            )

        self.assertEqual(summary.epochs_executed, 2)
        self.assertEqual(summary.best_epoch, 2)
        self.assertEqual(summary.best_validation_score, 0.55)
        self.assertEqual(predictor.model.weight.item(), 2.0)
        self.assertTrue(summary.stopped_early)

    def test_final_checkpoint_strategy_keeps_last_trained_state(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ),
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[
                    {"roc_auc": 0.90},
                    {"roc_auc": 0.80},
                    {"roc_auc": 0.70},
                ],
            ),
        ):
            summary = main.train_with_validation_selection(
                predictor,
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                training_policy=main.TrainingPolicy(
                    max_epochs=3,
                    selection_metric="roc_auc",
                    checkpoint_strategy="final",
                ),
                seed=4,
                device="cpu",
            )

        self.assertEqual(summary.best_epoch, 1)
        self.assertEqual(summary.selected_epoch, 3)
        self.assertEqual(summary.checkpoint_strategy, "final")
        self.assertEqual(summary.selected_loss, 3.0)
        self.assertEqual(predictor.model.weight.item(), 3.0)

    def test_early_stopping_patience_starts_after_minimum_epochs(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ),
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[
                    {"roc_auc": score}
                    for score in (0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
                ],
            ),
        ):
            summary = main.train_with_validation_selection(
                predictor,
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                training_policy=main.TrainingPolicy(
                    max_epochs=8,
                    selection_metric="roc_auc",
                    early_stopping_patience=2,
                    early_stopping_minimum_epochs=4,
                ),
                seed=4,
                device="cpu",
            )

        self.assertEqual(summary.epochs_executed, 6)
        self.assertEqual(summary.best_epoch, 1)
        self.assertEqual(summary.selected_epoch, 1)
        self.assertTrue(summary.stopped_early)
        self.assertEqual(predictor.model.weight.item(), 1.0)

    def test_training_policy_options_are_validated(self):
        predictor = SimpleNamespace(trainable=False)
        common = {
            "predictor": predictor,
            "train_data": Data(),
            "validation_data": Data(),
            "training_policy": main.TrainingPolicy(max_epochs=5),
            "seed": 1,
            "device": "cpu",
        }
        with self.assertRaisesRegex(ValueError, "checkpoint_strategy"):
            main.TrainingPolicy(max_epochs=5, checkpoint_strategy="")
        for minimum_epochs in (-1, 6, 1.5, True):
            with self.subTest(minimum_epochs=minimum_epochs):
                with self.assertRaisesRegex(ValueError, "minimum_epochs"):
                    main.TrainingPolicy(
                        max_epochs=5,
                        early_stopping_minimum_epochs=minimum_epochs,
                    )
        with self.assertRaisesRegex(ValueError, "epoch_callback"):
            main.train_with_validation_selection(**common, epoch_callback=object())

    def test_epoch_callback_receives_each_completed_epoch(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)

            def train(self, negative_edge_index=None):
                return torch.tensor(0.25)

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        records = []
        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ),
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[
                    {"roc_auc": 0.60, "average_precision": 0.55},
                    {"roc_auc": 0.70, "average_precision": 0.65},
                ],
            ),
        ):
            main.train_with_validation_selection(
                FakePredictor(),
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                training_policy=main.TrainingPolicy(
                    max_epochs=2,
                    selection_metric="roc_auc",
                ),
                seed=4,
                device="cpu",
                epoch_callback=records.append,
            )

        self.assertEqual(records, [
            {
                "epoch": 1,
                "training_loss": 0.25,
                "learning_rate": None,
                "validation_roc_auc": 0.60,
                "validation_average_precision": 0.55,
            },
            {
                "epoch": 2,
                "training_loss": 0.25,
                "learning_rate": None,
                "validation_roc_auc": 0.70,
                "validation_average_precision": 0.65,
            },
        ])

    @patch(
        "glpbench.runners.experiment.sample_training_negative_edges",
        wraps=link_prediction.sample_training_negative_edges,
    )
    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_unspecified_model_type_runs_both_models_on_both_splits(
        self, mock_load, mock_negative_sampling
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=None,
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            run_id = main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
        combinations = {(row["model_type"], row["split_type"]) for row in rows}
        self.assertEqual(combinations, {
            ("symmetric", "transductive"),
            ("symmetric", "inductive"),
            ("asymmetric", "transductive"),
            ("asymmetric", "inductive"),
            ("mlp", "transductive"),
            ("mlp", "inductive"),
            ("asymmetric-mlp", "transductive"),
            ("asymmetric-mlp", "inductive"),
            ("variational", "transductive"),
            ("variational", "inductive"),
            ("deal", "transductive"),
            ("deal", "inductive"),
            ("feature-cosine", "transductive"),
            ("feature-cosine", "inductive"),
            ("feature-mlp", "transductive"),
            ("feature-mlp", "inductive"),
        })
        self.assertEqual({row["run_id"] for row in rows}, {run_id})
        self.assertTrue(all(
            int(row["trainable_parameter_count"]) >= 0 for row in rows
        ))
        for model_type in {row["model_type"] for row in rows}:
            model_rows = [row for row in rows if row["model_type"] == model_type]
            self.assertEqual(
                len({row["trainable_parameter_count"] for row in model_rows}),
                1,
            )
        reduced_rows = [
            row for row in rows
            if row["model_type"] in {"asymmetric", "asymmetric-mlp"}
        ]
        self.assertTrue(all(
            float(row["reducer_fit_time_seconds"]) >= 0
            for row in reduced_rows
        ))
        self.assertEqual(
            len({row["reducer_fit_time_seconds"] for row in reduced_rows}),
            1,
        )
        self.assertTrue(all(
            row["reducer_fit_time_seconds"] == ""
            for row in rows
            if row["model_type"] not in {"asymmetric", "asymmetric-mlp"}
        ))
        trainable_rows = [
            row for row in rows if row["model_type"] != "feature-cosine"
        ]
        self.assertTrue(all(
            float(row["training_time_seconds"]) >= 0
            for row in trainable_rows
        ))
        self.assertEqual(
            {row["epochs_trained"] for row in trainable_rows}, {"1"}
        )
        self.assertEqual({row["best_epoch"] for row in trainable_rows}, {"1"})
        self.assertEqual(
            {row["selection_metric"] for row in trainable_rows}, {"roc_auc"}
        )
        for model_type in {
            row["model_type"] for row in trainable_rows
        }:
            model_runtimes = {
                row["training_time_seconds"]
                for row in trainable_rows
                if row["model_type"] == model_type
            }
            self.assertEqual(len(model_runtimes), 1)
        self.assertEqual(mock_negative_sampling.call_count, 7)
        self.assertEqual(
            [call.kwargs["seed"] for call in mock_negative_sampling.call_args_list],
            [args.seed] * 7,
        )
        cosine_rows = [
            row for row in rows if row["model_type"] == "feature-cosine"
        ]
        self.assertEqual(
            {row["trainable_parameter_count"] for row in cosine_rows}, {"0"}
        )
        self.assertEqual({row["conv_layer_type"] for row in cosine_rows}, {"None"})
        self.assertEqual({row["loss"] for row in cosine_rows}, {""})
        self.assertEqual(
            {row["training_time_seconds"] for row in cosine_rows}, {""}
        )
        self.assertEqual({row["epochs_trained"] for row in cosine_rows}, {"0"})
        self.assertEqual({row["best_epoch"] for row in cosine_rows}, {""})
        self.assertEqual({row["selection_metric"] for row in cosine_rows}, {""})
        self.assertEqual(
            {row["best_validation_score"] for row in cosine_rows}, {""}
        )
        self.assertEqual(
            {row["early_stopping_patience"] for row in cosine_rows}, {""}
        )
        self.assertEqual({row["stopped_early"] for row in cosine_rows}, {"False"})
        common_extra_fields = {
            "reduction_method", "validation_query_size", "learning_rate",
            "device", "query_selection", "validation_query_selection",
        }
        for row in rows:
            self.assertTrue(common_extra_fields.issubset(
                json.loads(row["extra_fields"])
            ))
        for model_type in ("mlp", "asymmetric-mlp"):
            mlp_config = json.loads(next(
                row["extra_fields"]
                for row in rows if row["model_type"] == model_type
            ))
            self.assertIn("decoder_hidden_channels", mlp_config)
            self.assertIn("decoder_dropout", mlp_config)
        variational_config = json.loads(next(
            row["extra_fields"]
            for row in rows if row["model_type"] == "variational"
        ))
        self.assertIn("kl_weight", variational_config)
        deal_rows = [row for row in rows if row["model_type"] == "deal"]
        self.assertEqual({row["conv_layer_type"] for row in deal_rows}, {"None"})
        deal_config = json.loads(deal_rows[0]["extra_fields"])
        self.assertEqual(deal_config["deal_theta_ss"], 1.0)
        self.assertEqual(deal_config["deal_lambda_sa"], 1.0)

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_explicit_model_type_list_runs_only_selected_models(self, mock_load):
        graph = self._graph()
        mock_load.return_value = _loaded_dataset(graph)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["asymmetric", "asymmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            with patch(
                "glpbench.preprocessing.adapters.fit_reducer",
                wraps=utils.fit_reducer,
            ) as mock_reduction:
                main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["model_type"] for row in rows}, {"asymmetric"})
        query_nodes = utils.select_query_node_sets(
            graph, query_size=args.query_size, seed=args.seed
        )
        fit_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        fit_mask[list(query_nodes.validation)] = False
        fit_mask[list(query_nodes.test)] = False
        expected_mean = graph.x[fit_mask].mean(dim=0).numpy()
        fitted_features = mock_reduction.call_args.args[0]
        fitted_mask = mock_reduction.call_args.kwargs["fit_nodes"]
        actual_mean = fitted_features[fitted_mask].mean(dim=0).numpy()
        self.assertTrue((actual_mean == expected_mean).all())
        for row in rows:
            extra_fields = json.loads(row["extra_fields"])
            self.assertEqual(
                extra_fields["reduction_method"], "pca"
            )
            self.assertEqual(
                extra_fields["validation_query_size"], 1
            )
            self.assertEqual(extra_fields["learning_rate"], 0.01)
            self.assertEqual(extra_fields["device"], "cpu")
            self.assertEqual(extra_fields["query_selection"], {
                "mode": "size", "value": 1, "resolved_size": 1,
            })
            self.assertEqual(extra_fields["validation_query_selection"], {
                "mode": "match-test", "value": None, "resolved_size": 1,
            })

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_gatv2_kwargs_are_passed_to_model_and_logged(self, mock_load):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GATv2Conv",
                gatv2_kwargs={"heads": 2, "concat": False, "dropout": 0.1},
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )
            with patch(
                "glpbench.runners.experiment.LinkPrediction", wraps=link_prediction.LinkPrediction
            ) as mock_predictor:
                main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(
            mock_predictor.call_args.kwargs["model_parameters"]["gatv2_kwargs"],
            args.gatv2_kwargs,
        )
        for row in rows:
            self.assertEqual(
                json.loads(row["extra_fields"])["gatv2_kwargs"],
                args.gatv2_kwargs,
            )

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_weight_decay_is_applied_to_non_asymmetric_trainable_models(
        self, mock_load
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                learning_rate=0.01,
                weight_decay=0.001,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            main.run_experiment(args)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual({float(row["weight_decay"]) for row in rows}, {0.001})

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_compact_progress_emits_model_seed_events(self, mock_load):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(Path(directory) / "results.csv"),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                learning_rate=0.01,
                weight_decay=0.0,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
                verbose=False,
            )
            events = []

            main.run_experiment(
                args,
                progress_callback=lambda event, **values: events.append(
                    (event, values)
                ),
                compact_progress=True,
            )

        event_names = [event for event, _ in events]
        self.assertEqual(event_names.count("run_started"), 1)
        self.assertEqual(event_names.count("run_completed"), 1)
        self.assertIn("splitting", {
            values.get("stage") for event, values in events
            if event == "stage"
        })
        self.assertIn("evaluating", {
            values.get("stage") for event, values in events
            if event == "stage"
        })

    @patch(
        "glpbench.runners.experiment.sample_training_negative_edges",
        wraps=link_prediction.sample_training_negative_edges,
    )
    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_asymmetric_run_expands_requested_regularizers(
        self, mock_load, mock_negative_sampling
    ):
        mock_load.return_value = _loaded_dataset(self._graph())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "regularization.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["asymmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                agae_regularization=["none", "latent-l2"],
                regularization_strength=0.1,
                learning_rate=0.005,
                weight_decay=0.01,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["regularization_type"] for row in rows},
            {"none", "latent-l2"},
        )
        self.assertEqual(
            {float(row["regularization_strength"]) for row in rows}, {0.1}
        )
        self.assertEqual({float(row["weight_decay"]) for row in rows}, {0.01})
        self.assertEqual(mock_negative_sampling.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in mock_negative_sampling.call_args_list],
            [args.seed, args.seed],
        )

