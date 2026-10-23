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


class TestPackageMigrationBoundary(unittest.TestCase):
    """Protect the installed package shell during the src-layout migration."""

    @staticmethod
    def _run_package_command(*arguments):
        project_root = _PROJECT_ROOT
        environment = os.environ.copy()
        source_path = str(project_root / "src")
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_path
            else os.pathsep.join((source_path, existing_path))
        )
        return subprocess.run(
            [sys.executable, "-m", "glpbench", *arguments],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_package_import_resolves_to_src_layout(self):
        project_root = _PROJECT_ROOT
        environment = os.environ.copy()
        source_path = str(project_root / "src")
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_path
            else os.pathsep.join((source_path, existing_path))
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import glpbench; "
                    "print(glpbench.__version__); "
                    "print(glpbench.__file__)"
                ),
            ],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output_lines = completed.stdout.strip().splitlines()
        project_configuration = (
            _PROJECT_ROOT / "pyproject.toml"
        ).read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', project_configuration)
        self.assertIn(
            'version = {attr = "glpbench.__version__"}',
            project_configuration,
        )
        self.assertRegex(output_lines[0], r"^\d+\.\d+\.\d+")
        self.assertTrue(
            output_lines[1].endswith("src/glpbench/__init__.py"),
            output_lines[1],
        )

    def test_top_level_cli_lists_current_subcommands(self):
        completed = self._run_package_command("--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for command in ("experiment", "benchmark", "search", "audit"):
            self.assertIn(command, completed.stdout)

    def test_cli_dispatches_each_current_subcommand(self):
        project_root = _PROJECT_ROOT
        source_path = str(project_root / "src")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        cli = importlib.import_module("glpbench.cli")
        expected_help = {
            "experiment": "Run a link-prediction experiment",
            "benchmark": "Run configuration-driven link-prediction benchmark grids",
            "search": "Run validation-only hyperparameter searches",
            "audit": "Run validation-only convergence audits",
        }

        for command, expected_text in expected_help.items():
            with self.subTest(command=command):
                output = StringIO()
                with (
                    patch.object(sys, "argv", ["glpbench", command, "--help"]),
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as exit_context,
                ):
                    cli.main()
                self.assertEqual(exit_context.exception.code, 0)
                self.assertIn(expected_text, output.getvalue())

    def test_readme_public_imports_remain_available(self):
        self.assertTrue(callable(datasets.load_dataset))
        self.assertTrue(callable(datasets.available_datasets))
        self.assertTrue(callable(utils.calculate_reduced_features))
        self.assertTrue(callable(utils.generate_comparable_splits))
        self.assertTrue(callable(utils.select_query_nodes))
        self.assertTrue(callable(link_prediction.LinkPrediction))
        self.assertTrue(callable(CSVResultLogger))
        self.assertTrue(callable(ExperimentResult))


