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


class TestExperimentLogging(unittest.TestCase):
    @staticmethod
    def _result(**overrides):
        values = {
            "dataset": "Cora",
            "model_type": "asymmetric",
            "split_type": "inductive",
            "conv_layer_type": "GCNConv",
            "hidden_channels": 128,
            "out_channels": 64,
            "query_size": 10,
            "train_ratio": 0.8,
            "epochs": 100,
            "random_seed": 42,
            "training_time_seconds": 12.5,
            "trainable_parameter_count": 1234,
            "reducer_fit_time_seconds": 0.25,
            "roc_auc": 0.91,
            "average_precision": 0.87,
        }
        values.update(overrides)
        return ExperimentResult(**values)

    def test_logger_creates_file_and_round_trips_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.csv"
            run_id = CSVResultLogger(path).log(
                self._result(extra_fields={"learning_rate": 0.01})
            )

            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], run_id)
            self.assertTrue(rows[0]["timestamp_utc"].endswith("+00:00"))
            self.assertEqual(rows[0]["dataset"], "Cora")
            self.assertEqual(float(rows[0]["roc_auc"]), 0.91)
            self.assertEqual(float(rows[0]["training_time_seconds"]), 12.5)
            self.assertEqual(int(rows[0]["trainable_parameter_count"]), 1234)
            self.assertEqual(float(rows[0]["reducer_fit_time_seconds"]), 0.25)
            self.assertEqual(rows[0]["regularization_type"], "none")
            self.assertEqual(float(rows[0]["regularization_strength"]), 0.0)
            self.assertEqual(float(rows[0]["weight_decay"]), 0.0)
            self.assertEqual(rows[0]["normalize_features"], "True")
            self.assertEqual(json.loads(rows[0]["extra_fields"]), {"learning_rate": 0.01})

    def test_configuration_id_is_stable_and_order_independent(self):
        first = make_configuration_id({
            "dataset": "Cora", "model_type": "asymmetric", "epochs": 100,
        })
        second = make_configuration_id({
            "epochs": 100, "model_type": "asymmetric", "dataset": "Cora",
        })
        changed = make_configuration_id({
            "dataset": "Cora", "model_type": "asymmetric", "epochs": 200,
        })

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_result_validates_asymmetric_regularization_fields(self):
        result = self._result(
            regularization_type="cosine-alignment",
            regularization_strength=0.1,
            weight_decay=0.001,
        )
        self.assertEqual(result.regularization_type, "cosine-alignment")

        with self.assertRaisesRegex(ValueError, "only supported by asymmetric"):
            self._result(
                model_type="symmetric",
                regularization_type="latent-l2",
                regularization_strength=0.1,
            )

    def test_logger_round_trips_checkpoint_selection_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            CSVResultLogger(path).log(self._result(
                epochs_trained=50,
                best_epoch=37,
                selected_epoch=37,
                training_policy="current",
                checkpoint_strategy="best-validation",
                selection_metric="roc_auc",
                best_validation_score=0.82,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                early_stopping_minimum_epochs=10,
                stopped_early=True,
                lr_scheduler="reduce-on-plateau",
                lr_scheduler_kwargs={
                    "factor": 0.5,
                    "patience": 5,
                    "threshold": 0.001,
                    "minimum_learning_rate": 1e-5,
                },
                final_learning_rate=0.00125,
            ))

            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))

        self.assertEqual(row["epochs_trained"], "50")
        self.assertEqual(row["best_epoch"], "37")
        self.assertEqual(row["selected_epoch"], "37")
        self.assertEqual(row["training_policy"], "current")
        self.assertEqual(row["checkpoint_strategy"], "best-validation")
        self.assertEqual(row["selection_metric"], "roc_auc")
        self.assertEqual(float(row["best_validation_score"]), 0.82)
        self.assertEqual(row["early_stopping_patience"], "10")
        self.assertEqual(float(row["early_stopping_min_delta"]), 0.001)
        self.assertEqual(row["early_stopping_minimum_epochs"], "10")
        self.assertEqual(row["stopped_early"], "True")
        self.assertEqual(row["lr_scheduler"], "reduce-on-plateau")
        self.assertEqual(
            json.loads(row["lr_scheduler_kwargs"])["factor"], 0.5
        )
        self.assertEqual(float(row["final_learning_rate"]), 0.00125)

    def test_epoch_history_logger_round_trips_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epochs.csv"
            logger = CSVEpochHistoryLogger(path, run_id="ablation-001")
            logger.log(EpochHistoryRecord(
                configuration_id="configuration-001",
                dataset="Cora",
                model_type="asymmetric",
                random_seed=42,
                training_policy="current",
                epoch=3,
                training_loss=0.25,
                learning_rate=0.005,
                validation_roc_auc=0.81,
                validation_average_precision=0.79,
            ))

            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))

        self.assertEqual(row["run_id"], "ablation-001")
        self.assertEqual(row["configuration_id"], "configuration-001")
        self.assertEqual(row["training_policy"], "current")
        self.assertEqual(row["epoch"], "3")
        self.assertEqual(float(row["validation_roc_auc"]), 0.81)

    def test_logger_appends_without_repeating_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path)
            first_run_id = logger.log(self._result(split_type="inductive"))
            second_run_id = logger.log(
                self._result(split_type="transductive", query_size=0)
            )

            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

            self.assertEqual(rows[0], CSVResultLogger.FIELDNAMES)
            self.assertEqual(len(rows), 3)
            run_id_index = CSVResultLogger.FIELDNAMES.index("run_id")
            self.assertEqual(first_run_id, logger.run_id)
            self.assertEqual(second_run_id, logger.run_id)
            self.assertEqual(rows[1][run_id_index], logger.run_id)
            self.assertEqual(rows[2][run_id_index], logger.run_id)

    def test_separate_loggers_generate_different_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            first_logger = CSVResultLogger(path)
            second_logger = CSVResultLogger(path)

            first_logger.log(self._result())
            second_logger.log(self._result())

            self.assertNotEqual(first_logger.run_id, second_logger.run_id)

    def test_logger_accepts_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path, run_id="experiment-001")

            returned_run_id = logger.log(self._result())

            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(returned_run_id, "experiment-001")
            self.assertEqual(row["run_id"], "experiment-001")

    def test_logger_rejects_empty_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CSVResultLogger(Path(directory) / "results.csv", run_id=" ")

    def test_logger_can_reset_to_generated_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path)
            original_run_id = logger.log(self._result())

            reset_run_id = logger.reset_run_id()
            logged_run_id = logger.log(self._result())

            self.assertNotEqual(reset_run_id, original_run_id)
            self.assertEqual(logged_run_id, reset_run_id)

    def test_logger_can_reset_to_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path)

            reset_run_id = logger.reset_run_id("experiment-002")
            logged_run_id = logger.log(self._result())

            self.assertEqual(reset_run_id, "experiment-002")
            self.assertEqual(logged_run_id, "experiment-002")
            with self.assertRaises(ValueError):
                logger.reset_run_id("")

    def test_logger_preserves_empty_optional_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            CSVResultLogger(path).log(self._result(roc_auc=None, average_precision=None))
            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["roc_auc"], "")
            self.assertEqual(row["average_precision"], "")

    def test_logger_rejects_mismatched_existing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                CSVResultLogger(path).log(self._result())

    def test_result_validates_parameters_and_metrics(self):
        invalid_values = [
            {"model_type": "unknown"},
            {"split_type": "unknown"},
            {"conv_layer_type": "unknown"},
            {"hidden_channels": 0},
            {"query_size": -1},
            {"normalize_features": "yes"},
            {"configuration_id": ""},
            {"train_ratio": 1.1},
            {"roc_auc": -0.1},
            {"loss": -1.0},
            {"training_time_seconds": -1.0},
            {"training_time_seconds": float("inf")},
            {"training_time_seconds": float("nan")},
            {"trainable_parameter_count": -1},
            {"trainable_parameter_count": 1.5},
            {"reducer_fit_time_seconds": -1.0},
            {"reducer_fit_time_seconds": float("inf")},
            {"epochs_trained": 101},
            {"epochs_trained": 10, "best_epoch": 11,
             "selection_metric": "roc_auc", "best_validation_score": 0.8},
            {"epochs_trained": 10, "selected_epoch": 11,
             "checkpoint_strategy": "final"},
            {"epochs_trained": 10, "selected_epoch": 5},
            {"checkpoint_strategy": "unknown"},
            {"training_policy": ""},
            {"epochs_trained": 10, "best_epoch": 5},
            {"selection_metric": "accuracy"},
            {"best_validation_score": 1.1},
            {"early_stopping_patience": 0},
            {"early_stopping_min_delta": -0.1},
            {"early_stopping_minimum_epochs": 101},
            {"lr_scheduler": "unknown"},
            {"lr_scheduler": "none", "lr_scheduler_kwargs": {"factor": 0.5}},
            {"final_learning_rate": -0.1},
            {"stopped_early": "yes"},
            {"epochs_trained": 50, "stopped_early": True},
            {"epochs_trained": 100, "early_stopping_patience": 2,
             "stopped_early": True},
        ]
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self._result(**values)

    def test_logger_requires_experiment_result(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                CSVResultLogger(Path(directory) / "results.csv").log({})


