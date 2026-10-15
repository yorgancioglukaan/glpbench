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


class TestLearningRateScheduling(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(
            lr_scheduling.validate_scheduler_config,
            training.validate_scheduler_config,
        )
        self.assertIs(
            lr_scheduling.build_lr_scheduler,
            training.build_lr_scheduler,
        )

    def test_registry_and_configuration_validation(self):
        self.assertEqual(lr_scheduling.available_lr_schedulers(), (
            "none", "reduce-on-plateau", "cosine-annealing"
        ))
        config = lr_scheduling.validate_scheduler_config(
            "reduce-on-plateau",
            {"factor": 0.5, "patience": 4, "minimum_learning_rate": 1e-5},
            epochs=100,
        )
        self.assertEqual(config["factor"], 0.5)
        self.assertEqual(config["patience"], 4)
        self.assertEqual(config["threshold"], 1e-4)
        with self.assertRaisesRegex(ValueError, "Unknown learning-rate"):
            lr_scheduling.validate_scheduler_config(
                "unknown", {}, epochs=100
            )
        with self.assertRaisesRegex(ValueError, "factor"):
            lr_scheduling.validate_scheduler_config(
                "reduce-on-plateau", {"factor": 1.0}, epochs=100
            )
        with self.assertRaisesRegex(ValueError, "does not accept"):
            lr_scheduling.validate_scheduler_config(
                "none", {"patience": 1}, epochs=100
            )

    def test_scheduler_step_uses_validation_only_for_plateau(self):
        plateau = unittest.mock.Mock()
        cosine = unittest.mock.Mock()

        lr_scheduling.step_lr_scheduler(
            plateau, "reduce-on-plateau", 0.75
        )
        lr_scheduling.step_lr_scheduler(cosine, "cosine-annealing", 0.75)

        plateau.step.assert_called_once_with(0.75)
        cosine.step.assert_called_once_with()

    def test_plateau_scheduler_updates_and_records_learning_rate(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=0.1
                )

            def train(self, negative_edge_index=None):
                self.optimizer.zero_grad()
                loss = self.model.weight.square().sum()
                loss.backward()
                self.optimizer.step()
                return loss.detach()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        records = []
        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ),
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[{"roc_auc": 0.8}] * 3,
            ),
        ):
            summary = main.train_with_validation_selection(
                predictor,
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                epochs=3,
                seed=4,
                selection_metric="roc_auc",
                device="cpu",
                lr_scheduler="reduce-on-plateau",
                lr_scheduler_kwargs={"factor": 0.5, "patience": 0},
                epoch_callback=records.append,
            )

        self.assertEqual(
            [record["learning_rate"] for record in records],
            [0.1, 0.1, 0.05],
        )
        self.assertAlmostEqual(summary.final_learning_rate, 0.025)
        self.assertEqual(summary.lr_scheduler, "reduce-on-plateau")
        self.assertEqual(summary.lr_scheduler_kwargs["factor"], 0.5)


