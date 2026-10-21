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


class TestFeatureReduction(unittest.TestCase):
    def test_public_exports_share_implementations(self):
        self.assertIs(
            feature_reduction.fit_reducer,
            preprocessing.fit_reducer,
        )
        self.assertIs(
            feature_reduction.transform_features,
            preprocessing.transform_features,
        )
        self.assertIs(
            feature_reduction._REDUCER_FACTORIES,
            preprocessing.reduction._REDUCER_FACTORIES,
        )
        self.assertTrue(callable(preprocessing.normalize_graph_features))

    class SpyReducer:
        def __init__(self, n_components, random_state=None, **kwargs):
            self.n_components = n_components
            self.random_state = random_state
            self.kwargs = kwargs
            self.fit_calls = []
            self.transform_calls = []

        def fit(self, features):
            self.fit_calls.append(features.copy())
            return self

        def transform(self, features):
            self.transform_calls.append(features.copy())
            return features[:, :self.n_components]

    def test_fit_uses_selected_nodes_and_transform_uses_all_nodes(self):
        features = torch.tensor([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ])
        fit_mask = torch.tensor([True, False, True, False])
        with patch.dict(
            feature_reduction._REDUCER_FACTORIES,
            {"spy": self.SpyReducer},
            clear=True,
        ):
            reducer = feature_reduction.fit_reducer(
                features,
                fit_nodes=fit_mask,
                method="spy",
                n_components=2,
                random_state=7,
                custom_option=True,
            )
            transformed = feature_reduction.transform_features(features, reducer)

        self.assertEqual(len(reducer.fit_calls), 1)
        self.assertTrue(
            torch.equal(torch.from_numpy(reducer.fit_calls[0]), features[fit_mask])
        )
        self.assertEqual(len(reducer.transform_calls), 1)
        self.assertTrue(
            torch.equal(torch.from_numpy(reducer.transform_calls[0]), features)
        )
        self.assertEqual(transformed.shape, (4, 2))
        self.assertEqual(reducer.random_state, 7)
        self.assertEqual(reducer.kwargs, {"custom_option": True})

    def test_update_uses_transform_without_refitting(self):
        data = Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        reducer = self.SpyReducer(n_components=1)
        reducer.fit(data.x.numpy())
        data.feature_reducer = reducer

        utils.update_reduced_features(data)

        self.assertEqual(len(reducer.fit_calls), 1)
        self.assertEqual(len(reducer.transform_calls), 1)
        self.assertTrue(torch.equal(data.x_reduced, data.x[:, :1]))

    def test_registry_and_invalid_method(self):
        self.assertEqual(feature_reduction.available_reduction_methods(), (
            "pca", "sparse-random-projection", "truncated-svd"
        ))
        with self.assertRaisesRegex(ValueError, "Unknown reduction method"):
            feature_reduction.fit_reducer(torch.ones(3, 2), method="unknown")

    def test_registered_reducers_transform_unseen_nodes(self):
        features = torch.tensor([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [1.0, 2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 1.0],
        ])
        fit_mask = torch.tensor([True, True, True, True, False, False])
        for method in ("truncated-svd", "sparse-random-projection"):
            with self.subTest(method=method):
                reducer = feature_reduction.fit_reducer(
                    features,
                    fit_nodes=fit_mask,
                    method=method,
                    n_components=2,
                    random_state=11,
                )
                transformed = feature_reduction.transform_features(
                    features, reducer
                )
                self.assertEqual(transformed.shape, (6, 2))
                self.assertTrue(torch.isfinite(transformed).all())
                self.assertEqual(reducer.reduction_method, method)

    def test_stochastic_reducers_are_reproducible(self):
        features = torch.arange(40, dtype=torch.float).reshape(8, 5)
        for method in ("truncated-svd", "sparse-random-projection"):
            with self.subTest(method=method):
                first = feature_reduction.fit_reducer(
                    features, method=method, n_components=2, random_state=5
                )
                second = feature_reduction.fit_reducer(
                    features, method=method, n_components=2, random_state=5
                )
                self.assertTrue(torch.allclose(
                    feature_reduction.transform_features(features, first),
                    feature_reduction.transform_features(features, second),
                ))

    def test_boolean_fit_mask_must_match_node_count(self):
        with self.assertRaisesRegex(ValueError, "one entry per node"):
            feature_reduction.fit_reducer(
                torch.ones(3, 2), fit_nodes=torch.tensor([True, False])
            )

