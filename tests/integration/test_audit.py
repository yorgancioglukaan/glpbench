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


class TestConvergenceAudit(unittest.TestCase):
    def test_project_audit_config_expands_all_dataset_model_pairs(self):
        config = convergence_audit.load_audit_config(
            "configs/convergence_audit.yaml"
        )
        jobs = convergence_audit.expand_audit_config(config)

        self.assertEqual(len(jobs), 8 * 6)
        self.assertEqual(config.benchmark_config.seeds, (400, 401, 402))
        self.assertTrue(all(job.parameters["epochs"] == 500 for job in jobs))
        self.assertTrue(all(
            job.parameters["checkpoint_strategy"] == "best-validation"
            and job.parameters["early_stopping_patience"] is None
            and job.parameters["lr_scheduler"] == "none"
            for job in jobs
        ))

    def test_dry_run_does_not_create_audit_artifacts(self):
        config = convergence_audit.load_audit_config(
            "configs/convergence_audit.yaml"
        )
        jobs = convergence_audit.expand_audit_config(config)[:2]
        with tempfile.TemporaryDirectory() as directory:
            isolated = replace(
                config,
                benchmark_config=replace(
                    config.benchmark_config,
                    output=str(Path(directory) / "results.csv"),
                ),
                epoch_history_output=str(Path(directory) / "epochs.csv"),
                failures_output=str(Path(directory) / "errors.jsonl"),
            )
            summary = convergence_audit.execute_audit(
                isolated, jobs, dry_run=True, run_id="audit-run"
            )

            self.assertEqual(summary.runs_total, 6)
            self.assertEqual(summary.runs_succeeded, 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_audit_config_rejects_test_or_early_stopping_options(self):
        document = yaml.safe_load(
            Path("configs/convergence_audit.yaml").read_text(encoding="utf-8")
        )
        document["training"]["early_stopping"] = {
            "enabled": True, "patience": 10
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "early stopping"):
                convergence_audit.load_audit_config(path)


if __name__ == "__main__":
    unittest.main()
