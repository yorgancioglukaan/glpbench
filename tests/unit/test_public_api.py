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
import glpbench.configuration as glp_configuration
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
from glpbench.configuration.loading import load_config as canonical_load_config


class TestLoggingPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(ExperimentResult, glp_logging.ExperimentResult)
        self.assertIs(CSVResultLogger, glp_logging.CSVResultLogger)
        self.assertIs(EpochHistoryRecord, glp_logging.EpochHistoryRecord)
        self.assertIs(
            CSVEpochHistoryLogger,
            glp_logging.CSVEpochHistoryLogger,
        )
        self.assertIs(
            make_configuration_id,
            glp_logging.make_configuration_id,
        )


class TestConfigurationPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(canonical_load_config, glp_configuration.load_config)
        self.assertTrue(callable(glp_configuration.resolve_experiment_specs))


class TestDatasetPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(datasets.load_dataset, glp_datasets.load_dataset)
        self.assertIs(
            datasets.available_datasets,
            glp_datasets.available_datasets,
        )
        self.assertTrue(callable(glp_datasets.get_dataset_provider))
        self.assertTrue(callable(glp_datasets.register_dataset_provider))
        self.assertTrue(callable(glp_datasets.validate_graph_data))


class TestModelPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(link_prediction.LinkPrediction, glp_models.LinkPrediction)
        self.assertIs(link_prediction.GAE_Model, glp_models.GAE_Model)
        self.assertIs(
            link_prediction.sample_training_negative_edges,
            glp_models.sample_training_negative_edges,
        )
        self.assertIs(
            link_prediction._MODEL_REGISTRY,
            glp_models.link_prediction._MODEL_REGISTRY,
        )


class TestProtocolPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(
            utils.generate_comparable_splits,
            glp_protocols.generate_comparable_splits,
        )
        self.assertIs(
            utils.select_query_node_sets,
            glp_protocols.select_query_node_sets,
        )
        self.assertIs(glp_protocols.SplitBundle, utils.SplitBundle)
        self.assertTrue(callable(glp_protocols.create_split_protocol))
        self.assertTrue(callable(glp_protocols.register_split_protocol))
        self.assertTrue(callable(glp_protocols.generate_transductive_splits))


class TestEvaluationPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(
            experiment_pipeline.prepare_candidate_data,
            glp_evaluation.prepare_candidate_data,
        )
        self.assertIs(
            experiment_pipeline.prepare_validation_data,
            glp_evaluation.prepare_validation_data,
        )


class TestExperimentRunnerPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(main.run_experiment, experiment_runner.run_experiment)
        self.assertIs(
            main.train_with_validation_selection,
            experiment_runner.train_with_validation_selection,
        )
        self.assertIs(main.calculate_metrics, experiment_runner.calculate_metrics)


class TestBenchmarkRunnerPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(
            benchmark.load_benchmark_config,
            benchmark_runner.load_benchmark_config,
        )
        self.assertIs(
            benchmark.execute_benchmark,
            benchmark_runner.execute_benchmark,
        )
        self.assertIs(benchmark.ExperimentJob, benchmark_runner.ExperimentJob)


class TestSearchRunnerPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(search.load_search_config, search_runner.load_search_config)
        self.assertIs(search.execute_search, search_runner.execute_search)
        self.assertIs(search.SearchCandidate, search_runner.SearchCandidate)


class TestAuditRunnerPublicExports(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(
            convergence_audit.load_audit_config,
            audit_runner.load_audit_config,
        )
        self.assertIs(
            convergence_audit.execute_audit,
            audit_runner.execute_audit,
        )
        self.assertIs(convergence_audit.AuditConfig, audit_runner.AuditConfig)


