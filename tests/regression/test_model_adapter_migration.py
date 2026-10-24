"""Regression coverage for behavior preserved by the adapter migration."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from glpbench.contracts import DatasetMetadata, LoadedDataset
from glpbench.models import (
    AGAEAdapter,
    AGAEMLPAdapter,
    DEALAdapter,
    FeatureCosineAdapter,
    FeatureMLPAdapter,
    GAEAdapter,
    GAEMLPAdapter,
    VGAEAdapter,
)
from glpbench.runners import experiment


ADAPTER_CASES = (
    (GAEAdapter, "gae"),
    (GAEMLPAdapter, "gae-mlp"),
    (AGAEAdapter, "agae"),
    (AGAEMLPAdapter, "agae-mlp"),
    (VGAEAdapter, "vgae"),
    (DEALAdapter, "deal"),
    (FeatureCosineAdapter, "feature-cosine"),
    (FeatureMLPAdapter, "feature-mlp"),
)


def _toy_data():
    x = torch.tensor([
        [1.0, 0.0, 0.2, 0.4],
        [0.0, 1.0, 0.3, 0.1],
        [0.4, 0.2, 1.0, 0.0],
        [0.7, 0.3, 0.2, 0.8],
        [0.1, 0.6, 0.9, 0.2],
        [0.8, 0.1, 0.4, 0.7],
    ])
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ], dtype=torch.long)
    candidates = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_label_index=candidates,
        edge_label=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    data.x_reduced = x[:, :3].clone()
    data.validation_query_mask = torch.tensor(
        [False, False, False, False, True, False]
    )
    data.test_query_mask = torch.tensor(
        [False, False, False, False, False, True]
    )
    data.visible_node_mask = ~(
        data.validation_query_mask | data.test_query_mask
    )
    data.split_setting = "transductive"
    return data


def _parameters(name):
    graph = {
        "hidden_channels": 5,
        "out_channels": 3,
        "conv_layer_type": "GCNConv",
        "gatv2_kwargs": {},
    }
    if name == "gae-mlp":
        return {**graph, "decoder_hidden_channels": 4, "decoder_dropout": 0.1}
    if name == "agae":
        return {
            **graph,
            "regularization_type": "cosine-alignment",
            "regularization_strength": 0.2,
        }
    if name == "agae-mlp":
        return {
            **graph,
            "decoder_hidden_channels": 4,
            "decoder_dropout": 0.1,
            "regularization_type": "cosine-alignment",
            "regularization_strength": 0.2,
        }
    if name == "vgae":
        return {**graph, "kl_weight": 0.3}
    if name == "deal":
        return {
            "hidden_channels": 5,
            "out_channels": 3,
            "theta_ss": 1.0,
            "theta_aa": 0.8,
            "theta_sa": 0.6,
            "lambda_ss": 1.0,
            "lambda_aa": 0.7,
            "lambda_sa": 0.5,
            "similarity_scale": 1.2,
            "similarity_margin": 0.1,
        }
    if name == "feature-mlp":
        return {"decoder_hidden_channels": 5, "decoder_dropout": 0.1}
    if name == "feature-cosine":
        return {}
    return graph


def _benchmark_graph():
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    directed = edges + [(target, source) for source, target in edges]
    return Data(
        x=torch.arange(32, dtype=torch.float).reshape(8, 4) + 1,
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
    )


class TestModelAdapterMigration(unittest.TestCase):
    def setUp(self):
        self.data = _toy_data()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def test_adapter_scores_match_wrapped_model_semantics(self):
        for adapter_class, name in ADAPTER_CASES:
            for setting in ("transductive", "semi-inductive"):
                with self.subTest(model=name, setting=setting):
                    data = self.data.clone()
                    data.split_setting = setting
                    if setting == "semi-inductive":
                        data.query_node_mask = data.test_query_mask.clone()
                        data.edge_label_index = torch.tensor(
                            [[0, 1], [5, 5]], dtype=torch.long
                        )
                    adapter = adapter_class()
                    torch.manual_seed(311)
                    model = adapter.build(data, _parameters(name))
                    model.eval()
                    with torch.no_grad():
                        representations = model.encode(data.x, data.edge_index)
                        expected = model.decode_edges(
                            representations, data.edge_label_index, data
                        ).view(-1)
                        if adapter.capabilities.output_kind != "probabilities":
                            expected = expected.sigmoid()
                    actual = adapter.score_edges(
                        model, data, data.edge_label_index
                    )
                    torch.testing.assert_close(actual, expected)

    def test_adapter_losses_match_wrapped_model_semantics(self):
        for adapter_class, name in ADAPTER_CASES:
            if not adapter_class().capabilities.trainable:
                continue
            with self.subTest(model=name):
                adapter = adapter_class()
                model = adapter.build(self.data, _parameters(name))
                reference = adapter.build(self.data, _parameters(name))
                reference.load_state_dict(model.state_dict())
                model.train()
                reference.train()
                torch.manual_seed(47)
                expected_total, expected_components = (
                    reference.compute_training_loss(
                        self.data,
                        self.data.edge_label_index,
                        self.data.edge_label,
                        self.criterion,
                    )
                )
                torch.manual_seed(47)
                actual = adapter.compute_training_loss(
                    model,
                    self.data,
                    self.data.edge_label_index,
                    self.data.edge_label,
                    self.criterion,
                )
                torch.testing.assert_close(actual.total, expected_total)
                self.assertEqual(set(actual.components), set(expected_components))
                for component, expected in expected_components.items():
                    torch.testing.assert_close(
                        actual.components[component], expected
                    )

    def test_training_resamples_negatives_once_per_executed_epoch(self):
        class Predictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                self.optimizer = None
                self.received = []

            def train(self, negative_edge_index=None):
                self.received.append(negative_edge_index.clone())
                return self.model.weight.sum().square()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        sampled_seeds = []

        def sample(_data, *, seed):
            sampled_seeds.append(seed)
            return torch.tensor([[seed], [seed + 1]], dtype=torch.long)

        predictor = Predictor()
        with (
            patch(
                "glpbench.runners.experiment.sample_training_negative_edges",
                side_effect=sample,
            ) as sampler,
            patch(
                "glpbench.runners.experiment.calculate_metrics",
                side_effect=[
                    {"roc_auc": 0.9},
                    {"roc_auc": 0.8},
                    {"roc_auc": 0.7},
                ],
            ),
        ):
            summary = experiment.train_with_validation_selection(
                predictor,
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                training_policy=experiment.TrainingPolicy(
                    max_epochs=5,
                    early_stopping_patience=2,
                ),
                seed=20,
                device="cpu",
            )

        self.assertEqual(summary.epochs_executed, 3)
        self.assertTrue(summary.stopped_early)
        self.assertEqual(sampler.call_count, 3)
        self.assertEqual(sampled_seeds, [20, 21, 22])
        self.assertEqual(len(predictor.received), 3)
        for index, negative_edges in enumerate(predictor.received):
            self.assertTrue(torch.equal(
                negative_edges,
                torch.tensor([[20 + index], [21 + index]]),
            ))

    @patch("glpbench.runners.experiment.load_registered_dataset")
    def test_adapter_metadata_preserves_logged_result_shape(self, mock_load):
        graph = _benchmark_graph()
        mock_load.return_value = LoadedDataset(
            graph,
            DatasetMetadata(
                name="Toy",
                family="Test",
                num_nodes=graph.num_nodes,
                num_edges=graph.num_edges,
                directed=False,
                has_node_features=True,
                num_node_features=graph.num_features,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            experiment.run_experiment(SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=[name for _, name in ADAPTER_CASES],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            ))
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 2 * len(ADAPTER_CASES))
        self.assertEqual(
            {(row["model_type"], row["split_type"]) for row in rows},
            {
                (name, setting)
                for _, name in ADAPTER_CASES
                for setting in ("transductive", "inductive")
            },
        )
        common_metadata = {
            "reduction_method", "validation_query_size", "learning_rate",
            "device", "query_selection", "validation_query_selection",
        }
        expected_model_metadata = {
            "gae": set(),
            "gae-mlp": {"decoder_hidden_channels", "decoder_dropout"},
            "agae": set(),
            "agae-mlp": {"decoder_hidden_channels", "decoder_dropout"},
            "vgae": {"kl_weight"},
            "deal": {
                "deal_theta_ss", "deal_theta_aa", "deal_theta_sa",
                "deal_lambda_ss", "deal_lambda_aa", "deal_lambda_sa",
                "deal_similarity_scale", "deal_similarity_margin",
            },
            "feature-cosine": set(),
            "feature-mlp": {"decoder_hidden_channels", "decoder_dropout"},
        }
        adapter_by_name = {
            name: adapter for adapter, name in ADAPTER_CASES
        }
        for row in rows:
            metadata = json.loads(row["extra_fields"])
            self.assertEqual(
                set(metadata).difference(common_metadata),
                expected_model_metadata[row["model_type"]],
            )
            self.assertEqual(row["dataset"], "Toy")
            self.assertEqual(row["random_seed"], "42")
            self.assertEqual(row["epochs"], "1")
            self.assertEqual(row["normalize_features"], "False")
            self.assertEqual(row["regularization_type"], "none")
            self.assertEqual(row["regularization_strength"], "0.0")
            self.assertEqual(row["weight_decay"], "0.0")
            expected_conv = (
                "GCNConv"
                if adapter_by_name[
                    row["model_type"]
                ].capabilities.uses_conv_layer
                else "None"
            )
            self.assertEqual(row["conv_layer_type"], expected_conv)


if __name__ == "__main__":
    unittest.main()
