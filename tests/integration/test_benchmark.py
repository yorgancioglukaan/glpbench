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


class TestBenchmarkRunner(unittest.TestCase):
    @staticmethod
    def _write_config(directory, content):
        path = Path(directory) / "benchmark.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _write_asymmetric_selection(directory):
        path = Path(directory) / "selected.yaml"
        path.write_text("""
version: 1
kind: selected-hyperparameters
selections:
  cora:
    agae:
      hyperparameters:
        hidden_channels: 32
        out_channels: 16
        learning_rate: 0.005
        weight_decay: 0.001
        normalize_features: false
        conv_layer_type: GCNConv
        reduction_method: truncated-svd
        regularization_type: cosine-alignment
        regularization_strength: 0.01
""", encoding="utf-8")
        return path

    def _write_training_ablation_config(self, directory):
        selected = self._write_asymmetric_selection(directory)
        output = Path(directory) / "results.csv"
        epochs = Path(directory) / "epochs.csv"
        config = self._write_config(directory, f"""
version: 1
kind: training-policy-ablation
output: {output}
error_output: {Path(directory) / "errors.jsonl"}
epoch_history_output: {epochs}
data_root: data
seeds: [7]
selected_configurations:
  path: {selected}
  granularity: [dataset, architecture]
  require_all_pairs: true
datasets: [Cora]
models: [agae]
splits:
  protocol: matched-semi-inductive
  query_fraction: 0.05
  validation_query_fraction: 0.05
  train_ratio: 0.8
  paired_across_policies: true
evaluation:
  split_types: [transductive, inductive]
  primary_split: inductive
  primary_metric: roc_auc
  secondary_metrics: [average_precision]
training:
  device: cpu
  selection_metric: roc_auc
  log_epoch_history: true
  epoch_history_metrics: [training_loss, validation_roc_auc]
training_policies:
  - name: current
    max_epochs: 10
    checkpoint: best-validation
    early_stopping:
      enabled: true
      patience: 2
      min_delta: 0.001
      minimum_epochs: 3
    scheduler:
      type: reduce-on-plateau
      factor: 0.5
      patience: 2
      threshold: 0.001
      minimum_learning_rate: 0.00001
  - name: full-budget-final
    max_epochs: 10
    checkpoint: final
    early_stopping:
      enabled: false
""")
        return config, output, epochs

    def test_load_and_expand_conditional_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, """
version: 1
output: out/test.csv
data_root: benchmark-data
seeds: [0, 1]
defaults:
  query_size: 5
  epochs: 2
  normalize_features: false
datasets: [Cora, CiteSeer]
models:
  - type: gae
    conv_layer_types: [GCNConv, SAGEConv]
  - type: agae
    conv_layer_types: [GCNConv]
    reduction_methods: [pca, truncated-svd]
    regularization:
      - {type: none, strength: 0.0}
      - {type: cosine-alignment, strength: 0.01}
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(config.datasets, ("cora", "citeseer"))
        self.assertEqual(config.seeds, (0, 1))
        self.assertEqual(len(jobs), 12)
        self.assertEqual(
            benchmark.describe_benchmark(config, jobs),
            {
                "datasets": 2,
                "jobs": 12,
                "seeds_per_job": 2,
                "model_runs": 24,
                "expected_result_rows": 48,
            },
        )
        first_args = jobs[0].to_namespace()
        self.assertEqual(first_args.dataset, "cora")
        self.assertEqual(first_args.seed, [0, 1])
        self.assertEqual(first_args.query_size, 5)
        self.assertFalse(first_args.normalize_features)

    def test_selected_configuration_expands_dataset_specific_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            selected_path = Path(directory) / "selected.yaml"
            selected_path.write_text("""
version: 1
kind: selected-hyperparameters
selections:
  cora:
    agae:
      hyperparameters:
        hidden_channels: 32
        out_channels: 16
        learning_rate: 0.005
        weight_decay: 0.001
        normalize_features: false
        conv_layer_type: SAGEConv
        reduction_method: truncated-svd
        regularization_type: cosine-alignment
        regularization_strength: 0.01
    deal:
      hyperparameters:
        hidden_channels: 64
        out_channels: 32
        learning_rate: 0.01
        weight_decay: 0.0001
        normalize_features: true
        conv_layer_type: GCNConv
        deal_profile: attribute-focused
        theta_ss: 0.1
        theta_aa: 1.0
        theta_sa: 0.5
        lambda_ss: 0.1
        lambda_aa: 1.0
        lambda_sa: 0.1
        similarity_scale: 1.0
        similarity_margin: 0.0
""", encoding="utf-8")
            path = self._write_config(directory, f"""
version: 2
kind: selected-hyperparameter-benchmark
output: out/preliminary.csv
seeds: [200, 201, 202]
selected_configurations:
  path: {selected_path}
  granularity: [dataset, architecture]
  require_all_pairs: true
defaults:
  query_fraction: 0.05
  validation_query_fraction: 0.05
  epochs: 100
datasets: [Cora]
models: [agae, deal]
""")

            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(config.seeds, (200, 201, 202))
        agae, deal = jobs
        self.assertEqual(agae.conv_layer_type, "SAGEConv")
        self.assertEqual(agae.reduction_method, "truncated-svd")
        self.assertEqual(agae.regularization.type, "cosine-alignment")
        self.assertEqual(agae.regularization.strength, 0.01)
        self.assertFalse(agae.parameters["normalize_features"])
        self.assertEqual(agae.parameters["hidden_channels"], 32)
        self.assertTrue(deal.parameters["normalize_features"])
        self.assertEqual(deal.parameters["deal_theta_ss"], 0.1)
        self.assertEqual(deal.parameters["deal_lambda_sa"], 0.1)
        self.assertEqual(
            benchmark.describe_benchmark(config, jobs)["expected_result_rows"],
            12,
        )

    def test_training_policy_ablation_expands_selected_model_per_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self._write_training_ablation_config(directory)
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(config.kind, "training-policy-ablation")
        self.assertTrue(config.log_epoch_history)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            {job.parameters["training_policy"] for job in jobs},
            {"current", "full-budget-final"},
        )
        current, final = jobs
        self.assertEqual(current.parameters["early_stopping_patience"], 2)
        self.assertEqual(current.parameters["early_stopping_minimum_epochs"], 3)
        self.assertEqual(current.parameters["checkpoint_strategy"], "best-validation")
        self.assertEqual(current.parameters["lr_scheduler"], "reduce-on-plateau")
        self.assertEqual(current.parameters["lr_scheduler_kwargs"]["factor"], 0.5)
        self.assertIsNone(final.parameters["early_stopping_patience"])
        self.assertEqual(final.parameters["checkpoint_strategy"], "final")
        self.assertEqual(final.parameters["lr_scheduler"], "none")
        self.assertEqual(
            benchmark.describe_benchmark(config, jobs)["expected_result_rows"],
            4,
        )

    def test_ablation_execution_logs_and_resumes_epoch_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, epoch_path = self._write_training_ablation_config(directory)
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

            def runner(args, **kwargs):
                args.epoch_callback({
                    "dataset": args.dataset,
                    "model_type": args.model_type[0],
                    "random_seed": args.seed[0],
                    "training_policy": args.training_policy,
                    "epoch": 1,
                    "training_loss": 0.25,
                    "learning_rate": 0.005,
                    "validation_roc_auc": 0.75,
                    "validation_average_precision": 0.70,
                })

            summary = benchmark.execute_benchmark(
                config, jobs, runner=runner, run_id="ablation-001"
            )
            with epoch_path.open(newline="", encoding="utf-8") as csv_file:
                first_rows = list(csv.DictReader(csv_file))

            benchmark.execute_benchmark(
                config, jobs, runner=runner, run_id="ablation-002", resume=True
            )
            with epoch_path.open(newline="", encoding="utf-8") as csv_file:
                resumed_rows = list(csv.DictReader(csv_file))

        self.assertEqual(summary.jobs_succeeded, 2)
        self.assertEqual(len(first_rows), 2)
        self.assertEqual(len(resumed_rows), 2)
        self.assertEqual(
            {row["training_policy"] for row in first_rows},
            {"current", "full-budget-final"},
        )
        self.assertEqual({row["run_id"] for row in first_rows}, {"ablation-001"})
        self.assertEqual(
            len({row["configuration_id"] for row in first_rows}), 2
        )

    def test_selected_configuration_requires_every_requested_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            selected_path = Path(directory) / "selected.yaml"
            selected_path.write_text("""
version: 1
kind: selected-hyperparameters
selections:
  cora: {}
""", encoding="utf-8")
            path = self._write_config(directory, f"""
version: 2
kind: selected-hyperparameter-benchmark
seeds: [200]
selected_configurations:
  path: {selected_path}
  granularity: [dataset, architecture]
  require_all_pairs: true
datasets: [Cora]
models: [gae]
""")
            config = benchmark.load_benchmark_config(path)

            with self.assertRaisesRegex(ValueError, "selections.cora.gae"):
                benchmark.expand_benchmark_config(config)

    def test_config_validation_rejects_incompatible_model_options(self):
        invalid_configs = (
            """
version: 1
seeds: [0]
datasets: [Cora]
models:
  - type: deal
    conv_layer_types: [GCNConv]
""",
            """
version: 1
seeds: [0]
datasets: [Cora]
models:
  - type: gae
    regularization:
      - {type: latent-l2, strength: 0.1}
""",
            """
version: 1
seeds: [0]
datasets: [unknown]
models:
  - type: gae
""",
            """
version: 1
seeds: [0]
datasets: [Cora]
models:
  - type: gae
    parameters:
      kl_weight: 0.5
""",
        )
        for content in invalid_configs:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = self._write_config(directory, content)
                with self.assertRaises(ValueError):
                    benchmark.load_benchmark_config(path)

    def test_expansion_rejects_invalid_resolved_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, """
version: 1
seeds: [0]
defaults:
  learning_rate: 0
datasets: [Cora]
models:
  - type: gae
""")
            config = benchmark.load_benchmark_config(path)

            with self.assertRaisesRegex(ValueError, "learning_rate"):
                benchmark.expand_benchmark_config(config)

    def test_asymmetric_mlp_grid_accepts_all_relevant_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, """
version: 1
seeds: [3]
datasets: [Cora]
models:
  - type: agae-mlp
    conv_layer_types: [GCNConv]
    reduction_methods: [pca, truncated-svd]
    regularization:
      - {type: none, strength: 0.0}
      - {type: cosine-alignment, strength: 0.1}
    parameters:
      decoder_hidden_channels: 16
      decoder_dropout: 0.2
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(
            job.model_type == "agae-mlp" for job in jobs
        ))
        self.assertEqual(
            {job.reduction_method for job in jobs}, {"pca", "truncated-svd"}
        )
        self.assertEqual(
            {job.regularization.type for job in jobs},
            {"none", "cosine-alignment"},
        )
        self.assertTrue(all(
            job.parameters["decoder_hidden_channels"] == 16 for job in jobs
        ))

    def test_dry_run_performs_no_execution_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            errors = Path(directory) / "errors.jsonl"
            path = self._write_config(directory, f"""
version: 1
output: {output}
error_output: {errors}
seeds: [0, 1]
datasets: [Cora]
models:
  - type: feature-cosine
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            runner = unittest.mock.Mock()

            summary = benchmark.execute_benchmark(
                config, jobs, dry_run=True, runner=runner, run_id="dry-run"
            )

            runner.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(errors.exists())
            self.assertEqual(summary.expected_result_rows, 4)

    def test_compact_progress_tracks_model_seed_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, f"""
version: 1
output: {Path(directory) / "results.csv"}
seeds: [3, 4]
datasets: [Cora]
models:
  - type: gae
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            received = []

            def runner(args, **kwargs):
                received.append((args, kwargs))
                callback = kwargs["progress_callback"]
                callback(
                    "run_started", dataset=args.dataset,
                    model_type="gae", seed=3, stage="training",
                )
                callback(
                    "run_completed", dataset=args.dataset,
                    model_type="gae", seed=3,
                )
                callback(
                    "run_started", dataset=args.dataset,
                    model_type="gae", seed=4, stage="training",
                )
                callback(
                    "run_skipped", dataset=args.dataset,
                    model_type="gae", seed=4,
                )

            progress = unittest.mock.Mock()
            with patch("glpbench.runners.benchmark.tqdm", return_value=progress) as mock_tqdm:
                summary = benchmark.execute_benchmark(
                    config, jobs, verbose=True, runner=runner
                )

        self.assertEqual(summary.jobs_succeeded, 1)
        self.assertTrue(received[0][0].verbose)
        self.assertTrue(received[0][1]["compact_progress"])
        self.assertEqual(progress.update.call_args_list, [
            unittest.mock.call(1), unittest.mock.call(1)
        ])
        mock_tqdm.assert_called_once()
        self.assertEqual(mock_tqdm.call_args.kwargs["total"], 2)
        self.assertEqual(mock_tqdm.call_args.kwargs["desc"], "Benchmark runs")

    def test_execution_shares_run_id_and_isolates_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = Path(directory) / "errors.jsonl"
            path = self._write_config(directory, f"""
version: 1
output: {Path(directory) / "results.csv"}
error_output: {errors}
seeds: [3]
datasets: [Cora]
models:
  - type: gae
  - type: feature-cosine
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            calls = []

            def runner(args, **kwargs):
                calls.append((args, kwargs))
                if args.model_type == ["gae"]:
                    raise RuntimeError("simulated failure")

            summary = benchmark.execute_benchmark(
                config, jobs, runner=runner, run_id="benchmark-001"
            )
            records = [
                json.loads(line)
                for line in errors.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary.jobs_succeeded, 1)
        self.assertEqual(summary.jobs_failed, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            {kwargs["run_id"] for _, kwargs in calls}, {"benchmark-001"}
        )
        self.assertIs(calls[0][1]["completed_results"], calls[1][1]["completed_results"])
        self.assertEqual(records[0]["error_type"], "RuntimeError")
        self.assertEqual(records[0]["message"], "simulated failure")

    def test_resume_loads_completed_configuration_seed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            row = {name: "" for name in CSVResultLogger.FIELDNAMES}
            row.update({
                "configuration_id": "config-1",
                "random_seed": "7",
            })
            with output.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=CSVResultLogger.FIELDNAMES
                )
                writer.writeheader()
                writer.writerow(row)
            path = self._write_config(directory, f"""
version: 1
output: {output}
seeds: [7]
datasets: [Cora]
models:
  - type: feature-cosine
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            received = []

            def runner(args, **kwargs):
                received.append(kwargs["completed_results"].copy())

            benchmark.execute_benchmark(
                config, jobs, resume=True, runner=runner
            )

        self.assertEqual(received, [{("config-1", 7)}])

    def test_fail_fast_records_and_reraises(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = Path(directory) / "errors.jsonl"
            path = self._write_config(directory, f"""
version: 1
error_output: {errors}
seeds: [0]
datasets: [Cora]
models:
  - type: gae
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

            with self.assertRaisesRegex(RuntimeError, "stop now"):
                benchmark.execute_benchmark(
                    config,
                    jobs,
                    fail_fast=True,
                    runner=unittest.mock.Mock(
                        side_effect=RuntimeError("stop now")
                    ),
                )
            self.assertTrue(errors.exists())


