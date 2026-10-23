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


class TestLinkPrediction(unittest.TestCase):
    @staticmethod
    def _training_data():
        return Data(
            x=torch.tensor([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [1.0, 1.0, 0.0],
                [0.5, 0.0, 1.0],
                [0.0, 0.5, 1.0],
            ]),
            x_reduced=torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 1.0],
                [1.0, 0.5],
            ]),
            edge_index=torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4],
                [1, 0, 2, 1, 3, 2, 4, 3],
            ]),
            edge_label_index=torch.tensor([[0, 1], [1, 2]]),
            edge_label=torch.ones(2),
        )

    @classmethod
    def _deal_training_data(cls):
        data = cls._training_data()
        data.edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ])
        data.validation_query_mask = torch.zeros(5, dtype=torch.bool)
        data.test_query_mask = torch.tensor(
            [False, False, False, False, True]
        )
        data.visible_node_mask = ~(
            data.validation_query_mask | data.test_query_mask
        )
        data.split_stage = "train"
        data.split_setting = "transductive"
        return data

    def test_model_construction_for_supported_convolutions(self):
        layer_types = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATv2Conv": GATv2Conv,
        }
        for model_class in (link_prediction.GAE_Model, link_prediction.AGAE_Model):
            for layer_name, layer_class in layer_types.items():
                with self.subTest(model=model_class.__name__, layer=layer_name):
                    model = model_class(3, 4, 2, conv_layer_type=layer_name)
                    self.assertIsInstance(model.conv1, layer_class)
                    self.assertIsInstance(model.conv2, layer_class)
                    self.assertIsInstance(model, link_prediction.LinkPredictionModel)

    def test_direct_model_constructors_default_to_gcn(self):
        model_classes = (
            link_prediction.GAE_Model,
            link_prediction.AGAE_Model,
            link_prediction.MLP_GAE_Model,
            link_prediction.MLP_AGAE_Model,
            link_prediction.VGAE_Model,
        )

        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                model = model_class(3, 4, 2)
                self.assertIsInstance(model.conv1, GCNConv)

    def test_models_share_data_aware_decoder_interface(self):
        data = self._training_data()
        symmetric = link_prediction.GAE_Model(3, 4, 2, conv_layer_type="GCNConv")
        asymmetric = link_prediction.AGAE_Model(3, 4, 2, conv_layer_type="GCNConv")
        z = torch.ones((data.num_nodes, 2))

        symmetric_scores = symmetric.decode_edges(z, data.edge_label_index, data)
        asymmetric_scores = asymmetric.decode_edges(z, data.edge_label_index, data)

        self.assertEqual(symmetric_scores.shape, data.edge_label.shape)
        self.assertEqual(asymmetric_scores.shape, data.edge_label.shape)
        self.assertEqual(symmetric.regularization_loss().item(), 0.0)
        self.assertEqual(asymmetric.regularization_loss().item(), 0.0)

    def test_invalid_model_and_convolution_types(self):
        data = self._training_data()
        with self.assertRaises(ValueError):
            link_prediction.GAE_Model(3, 4, 2, conv_layer_type="InvalidConv")
        with self.assertRaises(ValueError):
            link_prediction.AGAE_Model(3, 4, 2, conv_layer_type="InvalidConv")
        with self.assertRaises(ValueError):
            link_prediction.LinkPrediction(data, model_type="invalid")

    def test_available_model_types_come_from_registry(self):
        self.assertEqual(
            link_prediction.available_model_types(),
            (
                "symmetric", "asymmetric", "mlp", "asymmetric-mlp", "variational",
                "deal", "feature-cosine", "feature-mlp",
            ),
        )

    def test_model_registrations_describe_asymmetric_mlp_capabilities(self):
        spec = link_prediction.get_model_registration("asymmetric-mlp")

        self.assertIs(spec.constructor, link_prediction.MLP_AGAE_Model)
        self.assertTrue(spec.capabilities.uses_conv_layer)
        self.assertTrue(spec.capabilities.requires_reduced_features)
        self.assertTrue(spec.capabilities.supports_agae_regularization)
        self.assertTrue(spec.capabilities.uses_mlp_decoder)
        with self.assertRaisesRegex(ValueError, "Unexpected model type"):
            link_prediction.get_model_registration("unknown")

    def test_deal_has_structural_embeddings_only_for_seen_nodes(self):
        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="deal", hidden_channels=4, out_channels=2
        )

        structure, attributes = predictor.model.encode(data.x, data.edge_index)

        self.assertEqual(predictor.model.structure_embeddings.num_embeddings, 4)
        self.assertEqual(predictor.model.global_to_visible[4].item(), -1)
        self.assertTrue(torch.equal(structure[4], torch.zeros(2)))
        self.assertEqual(attributes.shape, (data.num_nodes, 2))

    def test_deal_training_uses_three_loss_components(self):
        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="deal", hidden_channels=4, out_channels=2
        )
        negatives = torch.tensor([[0, 3], [2, 0]])
        before = [
            parameter.detach().clone() for parameter in predictor.model.parameters()
        ]

        loss = predictor.train(negative_edge_index=negatives)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(predictor.last_loss_components),
            {"structure", "attribute", "alignment"},
        )
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, predictor.model.parameters())
        ))

    def test_deal_uses_split_specific_score_components(self):
        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="deal",
            hidden_channels=4,
            out_channels=2,
            lambda_ss=2.0,
            lambda_aa=3.0,
            lambda_sa=5.0,
        )
        predictor.model.eval()
        representations = predictor.model.encode(data.x, data.edge_index)

        transductive = data.clone()
        transductive.split_setting = "transductive"
        transductive.edge_label_index = torch.tensor([[0], [3]])
        trans_components = predictor.model._component_logits(
            representations, transductive.edge_label_index
        )
        expected_trans = sum(
            weight * score
            for weight, score in zip((2.0, 3.0, 5.0), trans_components)
        ) / 10.0

        inductive = data.clone()
        inductive.split_setting = "semi-inductive"
        inductive.edge_label_index = torch.tensor([[0], [4]])
        ind_components = predictor.model._component_logits(
            representations, inductive.edge_label_index
        )
        expected_ind = (
            3.0 * ind_components[1] + 5.0 * ind_components[2]
        ) / 8.0

        self.assertTrue(torch.allclose(
            predictor.test(transductive), expected_trans.sigmoid()
        ))
        self.assertTrue(torch.allclose(
            predictor.test(inductive), expected_ind.sigmoid()
        ))

        validation = inductive.clone()
        validation.split_stage = "validation"
        validation.split_setting = "semi-inductive"
        self.assertTrue(torch.allclose(
            predictor.test(validation), expected_ind.sigmoid()
        ))

    def test_deal_rejects_missing_split_metadata_and_query_training_edges(self):
        with self.assertRaisesRegex(ValueError, "validation_query_mask"):
            link_prediction.LinkPrediction(
                self._training_data(), model_type="deal"
            )

        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(data, model_type="deal")
        with self.assertRaisesRegex(ValueError, "only visible nodes"):
            predictor.train(negative_edge_index=torch.tensor([[0], [4]]))

    def test_feature_cosine_scores_features_without_training(self):
        data = Data(
            x=torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
            ]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_label_index=torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]]),
        )
        predictor = link_prediction.LinkPrediction(
            data, model_type="feature-cosine"
        )

        probabilities = predictor.test(data)

        self.assertTrue(torch.allclose(
            probabilities, torch.tensor([1.0, 0.5, 0.0, 0.5])
        ))
        self.assertFalse(predictor.trainable)
        self.assertIsNone(predictor.optimizer)
        with self.assertRaisesRegex(RuntimeError, "not trainable"):
            predictor.train()

    def test_feature_cosine_is_symmetric_and_ignores_message_passing_edges(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="feature-cosine"
        )
        original = predictor.test(data)
        reversed_data = data.clone()
        reversed_data.edge_label_index = data.edge_label_index.flip(0)
        reversed_data.edge_index = torch.empty((2, 0), dtype=torch.long)

        reversed_scores = predictor.test(reversed_data)

        self.assertTrue(torch.allclose(original, reversed_scores))
        self.assertTrue(torch.isfinite(original).all())
        self.assertTrue(((original >= 0) & (original <= 1)).all())

    def test_feature_mlp_is_symmetric_trainable_and_graph_independent(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="feature-mlp",
            decoder_hidden_channels=5,
            decoder_dropout=0.0,
        )
        original_logits = predictor.model.decode_edges(
            data.x, data.edge_label_index, data
        )
        reversed_logits = predictor.model.decode_edges(
            data.x, data.edge_label_index.flip(0), data
        )
        graphless = data.clone()
        graphless.edge_index = torch.empty((2, 0), dtype=torch.long)
        original_probabilities = predictor.test(data)
        graphless_probabilities = predictor.test(graphless)
        parameters_before = [
            parameter.detach().clone()
            for parameter in predictor.model.parameters()
        ]

        loss = predictor.train(data)

        self.assertTrue(predictor.trainable)
        self.assertTrue(torch.allclose(original_logits, reversed_logits))
        self.assertTrue(torch.allclose(
            original_probabilities, graphless_probabilities
        ))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(
                parameters_before, predictor.model.parameters()
            )
        ))

    def test_feature_mlp_spec_and_configuration(self):
        data = self._training_data()
        spec = link_prediction.get_model_registration("feature-mlp")

        self.assertIs(spec.constructor, link_prediction.FeatureMLPModel)
        self.assertFalse(spec.capabilities.uses_conv_layer)
        self.assertTrue(spec.capabilities.uses_mlp_decoder)
        with self.assertRaisesRegex(ValueError, "decoder_hidden_channels"):
            link_prediction.LinkPrediction(
                data, model_type="feature-mlp", decoder_hidden_channels=0
            )
        with self.assertRaisesRegex(ValueError, "decoder_dropout"):
            link_prediction.LinkPrediction(
                data, model_type="feature-mlp", decoder_dropout=1.0
            )

    def test_mlp_decoder_is_symmetric_and_trainable(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="mlp",
            conv_layer_type="GCNConv",
            hidden_channels=4,
            out_channels=2,
            decoder_hidden_channels=5,
            decoder_dropout=0.0,
        )
        z = predictor.model.encode(data.x, data.edge_index)
        reversed_edges = data.edge_label_index.flip(0)

        scores = predictor.model.decode(z, data.edge_label_index)
        reversed_scores = predictor.model.decode(z, reversed_edges)
        decoder_before = [
            parameter.detach().clone()
            for parameter in predictor.model.edge_decoder.parameters()
        ]
        loss = predictor.train(data)

        self.assertTrue(torch.allclose(scores, reversed_scores))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(
                decoder_before, predictor.model.edge_decoder.parameters()
            )
        ))

    def test_mlp_decoder_validates_configuration(self):
        data = self._training_data()
        for model_type in ("mlp", "asymmetric-mlp", "feature-mlp"):
            with self.subTest(model_type=model_type):
                with self.assertRaises(ValueError):
                    link_prediction.LinkPrediction(
                        data,
                        model_type=model_type,
                        decoder_hidden_channels=0,
                    )
                with self.assertRaises(ValueError):
                    link_prediction.LinkPrediction(
                        data, model_type=model_type, decoder_dropout=1.0
                    )

    def test_asymmetric_mlp_uses_latent_sources_and_feature_targets(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric-mlp",
            conv_layer_type="GCNConv",
            hidden_channels=4,
            out_channels=2,
            decoder_hidden_channels=5,
            decoder_dropout=0.0,
        )
        z = predictor.model.encode(data.x, data.edge_index)
        edges = data.edge_label_index
        source = z[edges[0]]
        target = data.x_reduced[edges[1]]
        expected_features = torch.cat(
            [source * target, torch.abs(source - target)], dim=-1
        )

        scores = predictor.model.decode_edges(z, edges, data)

        self.assertTrue(torch.allclose(
            scores,
            predictor.model.edge_decoder(expected_features).squeeze(-1),
        ))
        self.assertEqual(scores.shape, data.edge_label.shape)

    def test_asymmetric_mlp_decoder_is_trainable_and_regularized(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric-mlp",
            hidden_channels=4,
            out_channels=2,
            decoder_hidden_channels=5,
            regularization_type="latent-l2",
            regularization_strength=0.1,
        )
        decoder_before = [
            parameter.detach().clone()
            for parameter in predictor.model.edge_decoder.parameters()
        ]

        loss = predictor.train(data)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(predictor.last_regularization_loss.item(), 0.0)
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(
                decoder_before, predictor.model.edge_decoder.parameters()
            )
        ))

    def test_variational_model_samples_trains_and_infers_deterministically(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="variational",
            conv_layer_type="GCNConv",
            hidden_channels=4,
            out_channels=2,
            kl_weight=0.5,
        )

        predictor.model.train()
        first_sample = predictor.model.encode(data.x, data.edge_index)
        second_sample = predictor.model.encode(data.x, data.edge_index)
        self.assertEqual(predictor.model.mu.shape, (data.num_nodes, 2))
        self.assertEqual(predictor.model.logstd.shape, (data.num_nodes, 2))
        self.assertFalse(torch.equal(first_sample, second_sample))

        loss = predictor.train(data)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(predictor.last_regularization_loss.item(), 0.0)
        self.assertTrue(torch.allclose(
            loss,
            predictor.last_reconstruction_loss + predictor.last_regularization_loss,
        ))

        predictor.model.eval()
        first_mean = predictor.model.encode(data.x, data.edge_index)
        second_mean = predictor.model.encode(data.x, data.edge_index)
        self.assertTrue(torch.equal(first_mean, second_mean))
        self.assertTrue(torch.equal(first_mean, predictor.model.mu))

    def test_variational_kl_weight_scales_regularization(self):
        data = self._training_data()
        model = link_prediction.VGAE_Model(
            3, 4, 2, conv_layer_type="GCNConv", kl_weight=1.0
        )
        model.encode(data.x, data.edge_index)
        base_loss = model.regularization_loss()
        model.kl_weight = 2.0
        self.assertTrue(torch.allclose(model.regularization_loss(), 2 * base_loss))
        with self.assertRaises(ValueError):
            link_prediction.VGAE_Model(
                3, 4, 2, conv_layer_type="GCNConv", kl_weight=-1
            )

    def test_new_baselines_support_all_convolution_types(self):
        data = self._training_data()
        for model_type in ("mlp", "asymmetric-mlp", "variational"):
            for conv_layer_type in (
                "GCNConv", "GraphConv", "SAGEConv", "GATv2Conv"
            ):
                with self.subTest(
                    model_type=model_type, conv_layer_type=conv_layer_type
                ):
                    predictor = link_prediction.LinkPrediction(
                        data,
                        model_type=model_type,
                        conv_layer_type=conv_layer_type,
                        hidden_channels=4,
                        out_channels=2,
                    )
                    predictor.model.eval()
                    encoded = predictor.model.encode(data.x, data.edge_index)
                    scores = predictor.model.decode_edges(
                        encoded, data.edge_label_index, data
                    )
                    self.assertEqual(encoded.shape, (data.num_nodes, 2))
                    self.assertEqual(scores.shape, data.edge_label.shape)

    def test_encoder_output_shapes_are_finite(self):
        data = self._training_data()
        for model_class in (link_prediction.GAE_Model, link_prediction.AGAE_Model):
            for layer_name in ("GCNConv", "GraphConv", "SAGEConv", "GATv2Conv"):
                with self.subTest(model=model_class.__name__, layer=layer_name):
                    model = model_class(3, 4, 2, conv_layer_type=layer_name)
                    encoded = model.encode(data.x, data.edge_index)
                    self.assertEqual(encoded.shape, (data.num_nodes, 2))
                    self.assertTrue(torch.isfinite(encoded).all())

    def test_symmetric_decoder_matches_dot_products_and_is_symmetric(self):
        model = link_prediction.GAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]])
        edges = torch.tensor([[0, 1, 2], [1, 0, 1]])

        scores = model.decode(z, edges)

        expected = torch.tensor([11.0, 11.0, 5.0])
        self.assertTrue(torch.equal(scores, expected))
        self.assertEqual(scores[0].item(), scores[1].item())

    def test_asymmetric_decoder_matches_source_target_dot_products(self):
        model = link_prediction.AGAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_reduced = torch.tensor([[2.0, 0.0], [0.0, 5.0]])
        edges = torch.tensor([[0, 1], [1, 0]])

        scores = model.decode(z, edges, x_reduced)

        self.assertTrue(torch.equal(scores, torch.tensor([10.0, 6.0])))
        self.assertNotEqual(scores[0].item(), scores[1].item())

    def test_available_agae_regularizers_come_from_registry(self):
        self.assertEqual(
            link_prediction.available_agae_regularizers(),
            (
                "none", "latent-l2", "mse-alignment",
                "cosine-alignment", "graph-smoothness",
            ),
        )

    def test_agae_regularizers_are_finite_and_nonnegative(self):
        data = self._deal_training_data()
        z = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.0],
            [1000.0, -1000.0],
        ])
        model = link_prediction.AGAE_Model(3, 4, 2)

        for name, regularizer in link_prediction._AGAE_REGULARIZERS.items():
            with self.subTest(regularizer=name):
                penalty = regularizer(model, data, z)
                self.assertEqual(penalty.ndim, 0)
                self.assertTrue(torch.isfinite(penalty))
                self.assertGreaterEqual(penalty.item(), 0.0)

    def test_agae_node_regularizers_exclude_query_nodes(self):
        data = self._deal_training_data()
        z = torch.ones((data.num_nodes, 2))
        baseline = {
            name: link_prediction._AGAE_REGULARIZERS[name](None, data, z)
            for name in ("latent-l2", "mse-alignment", "cosine-alignment")
        }
        z[4] = torch.tensor([1e6, -1e6])
        data.x_reduced[4] = torch.tensor([-1e6, 1e6])

        for name, expected in baseline.items():
            with self.subTest(regularizer=name):
                actual = link_prediction._AGAE_REGULARIZERS[name](None, data, z)
                self.assertTrue(torch.allclose(actual, expected))

    def test_agae_regularization_strength_scales_only_penalty(self):
        data = self._deal_training_data()
        model = link_prediction.AGAE_Model(
            3, 4, 2,
            regularization_type="latent-l2",
            regularization_strength=1.0,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        edges = torch.tensor([[0, 1, 0, 3], [1, 2, 2, 0]])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

        first_total, first_components = model.compute_training_loss(
            data, edges, labels, criterion
        )
        model.regularization_strength = 2.0
        second_total, second_components = model.compute_training_loss(
            data, edges, labels, criterion
        )

        self.assertTrue(torch.allclose(
            second_components["regularization"],
            2 * first_components["regularization"],
        ))
        self.assertTrue(torch.allclose(
            first_components["reconstruction"],
            second_components["reconstruction"],
        ))
        self.assertTrue(torch.allclose(
            first_total,
            first_components["reconstruction"]
            + first_components["regularization"],
        ))
        self.assertTrue(torch.allclose(
            second_total,
            second_components["reconstruction"]
            + second_components["regularization"],
        ))

    def test_agae_regularization_and_optimizer_configuration_are_validated(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric",
            learning_rate=0.005,
            weight_decay=0.01,
        )
        self.assertEqual(predictor.optimizer.param_groups[0]["lr"], 0.005)
        self.assertEqual(predictor.optimizer.param_groups[0]["weight_decay"], 0.01)

        invalid_configs = (
            {"regularization_type": "unknown"},
            {"regularization_strength": -1.0},
            {"learning_rate": 0.0},
            {"weight_decay": -1.0},
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    link_prediction.LinkPrediction(
                        data, model_type="asymmetric", **config
                    )

    def test_decode_all_applies_score_threshold(self):
        symmetric = link_prediction.GAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        asymmetric = link_prediction.AGAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        x_reduced = torch.tensor([[0.4, 0.0], [0.0, 0.6]])

        self.assertTrue(torch.equal(
            symmetric.decode_all(z), torch.tensor([[0, 1], [0, 1]])
        ))
        self.assertTrue(torch.equal(
            asymmetric.decode_all(z, x_reduced), torch.tensor([[1], [1]])
        ))

    def test_link_prediction_constructor_configuration(self):
        data = self._training_data()
        symmetric = link_prediction.LinkPrediction(
            data, model_type="symmetric", conv_layer_type="GraphConv",
            hidden_channels=7, out_channels=2,
        )
        asymmetric = link_prediction.LinkPrediction(
            data, model_type="asymmetric", conv_layer_type="SAGEConv",
            hidden_channels=6, out_channels=2,
        )

        self.assertIsInstance(symmetric.model, link_prediction.GAE_Model)
        self.assertIsInstance(symmetric.model.conv1, GraphConv)
        self.assertEqual(symmetric.model.conv1.out_channels, 7)
        self.assertEqual(symmetric.model.conv2.out_channels, 2)
        self.assertIsInstance(asymmetric.model, link_prediction.AGAE_Model)
        self.assertIsInstance(asymmetric.model.conv1, SAGEConv)
        self.assertEqual(asymmetric.model.conv1.out_channels, 6)
        self.assertIsInstance(symmetric.criterion, torch.nn.BCEWithLogitsLoss)
        optimizer_params = {id(param) for group in symmetric.optimizer.param_groups
                            for param in group["params"]}
        self.assertEqual(optimizer_params, {id(param) for param in symmetric.model.parameters()})

    def test_default_constructor_uses_symmetric_gcn(self):
        predictor = link_prediction.LinkPrediction(self._training_data())
        self.assertIsInstance(predictor.model, link_prediction.GAE_Model)
        self.assertIsInstance(predictor.model.conv1, GCNConv)

    def test_gatv2_options_are_forwarded_and_output_shape_is_preserved(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric",
            conv_layer_type="GATv2Conv",
            hidden_channels=4,
            out_channels=2,
            heads=3,
            concat=True,
            dropout=0.2,
            share_weights=True,
        )

        self.assertIsInstance(predictor.model.conv1, GATv2Conv)
        self.assertEqual(predictor.model.conv1.heads, 3)
        self.assertTrue(predictor.model.conv1.concat)
        self.assertEqual(predictor.model.conv1.dropout, 0.2)
        self.assertTrue(predictor.model.conv1.share_weights)
        self.assertEqual(predictor.model.conv2.heads, 1)
        self.assertFalse(predictor.model.conv2.concat)
        encoded = predictor.model.encode(data.x, data.edge_index)
        self.assertEqual(encoded.shape, (data.num_nodes, 2))
        loss = predictor.train(data)
        probabilities = predictor.test(data)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(probabilities.shape, (data.edge_label_index.size(1),))

    def test_gatv2_options_can_be_passed_as_dictionary(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="symmetric",
            conv_layer_type="GATv2Conv",
            hidden_channels=4,
            out_channels=2,
            gatv2_kwargs={"heads": 2, "concat": False, "dropout": 0.1},
        )

        self.assertEqual(predictor.model.conv1.heads, 2)
        self.assertFalse(predictor.model.conv1.concat)
        self.assertEqual(predictor.model.conv1.dropout, 0.1)

    def test_gatv2_kwargs_must_be_dictionary(self):
        with self.assertRaisesRegex(TypeError, "gatv2_kwargs"):
            link_prediction.LinkPrediction(
                self._training_data(),
                conv_layer_type="GATv2Conv",
                gatv2_kwargs="invalid",
            )

    @patch("glpbench.models.link_prediction.negative_sampling")
    def test_asymmetric_training_step_updates_parameters(self, mock_sampling):
        data = self._training_data()
        mock_sampling.return_value = torch.tensor([[0, 4], [3, 1]])
        predictor = link_prediction.LinkPrediction(
            data, model_type="asymmetric", hidden_channels=4, out_channels=2
        )
        before = [parameter.detach().clone() for parameter in predictor.model.parameters()]

        loss = predictor.train(data)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            not torch.equal(old, new) for old, new in zip(before, predictor.model.parameters())
        ))
        self.assertTrue(torch.allclose(
            loss,
            predictor.last_reconstruction_loss + predictor.last_regularization_loss,
        ))
        self.assertEqual(predictor.last_regularization_loss.item(), 0.0)
        mock_sampling.assert_called_once_with(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1),
        )
        sampled = mock_sampling.return_value
        positives = {tuple(edge) for edge in data.edge_index.t().tolist()}
        self.assertEqual(sampled.size(1), data.edge_label_index.size(1))
        self.assertTrue(all(tuple(edge) not in positives for edge in sampled.t().tolist()))

    def test_train_without_argument_uses_constructor_data(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="asymmetric", hidden_channels=4, out_channels=2
        )
        loss = predictor.train()
        self.assertTrue(torch.isfinite(loss))

    def test_train_accepts_precomputed_negative_edges(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        negative_edges = torch.tensor([[0, 4], [3, 1]])

        with patch(
            "glpbench.models.link_prediction.negative_sampling"
        ) as mock_sampling:
            loss = predictor.train(
                data, negative_edge_index=negative_edges
            )

        self.assertTrue(torch.isfinite(loss))
        mock_sampling.assert_not_called()

    def test_train_validates_precomputed_negative_edges(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        invalid_edges = (
            torch.tensor([0, 1]),
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[0], [data.num_nodes]]),
        )
        for edge_index in invalid_edges:
            with self.subTest(edge_index=edge_index):
                with self.assertRaises((TypeError, ValueError)):
                    predictor.train(
                        data, negative_edge_index=edge_index
                    )

    def test_seeded_negative_sampling_is_reproducible_and_rng_safe(self):
        data = self._training_data()
        torch.manual_seed(19)
        expected_next_random = torch.rand(1)
        torch.manual_seed(19)
        random.seed(19)
        expected_next_python_random = random.random()
        random.seed(19)

        first = link_prediction.sample_training_negative_edges(data, seed=7)
        actual_next_random = torch.rand(1)
        actual_next_python_random = random.random()
        second = link_prediction.sample_training_negative_edges(data, seed=7)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(actual_next_random, expected_next_random))
        self.assertEqual(actual_next_python_random, expected_next_python_random)

    def test_training_negative_sampling_excludes_query_nodes(self):
        data = self._deal_training_data()

        negatives = link_prediction.sample_training_negative_edges(data, seed=7)

        self.assertTrue(bool(data.visible_node_mask[negatives].all()))

    def test_graph_context_model_can_override_training_loss(self):
        class ContextModel(link_prediction.LinkPredictionModel):
            def __init__(
                self, in_channels, hidden_channels, out_channels, train_data,
                **kwargs
            ):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.constructor_data = train_data
                self.received_edges = None
                self.received_labels = None

            def decode_edges(self, z, edge_label_index, data):
                raise AssertionError("custom training hook should be used")

            def compute_training_loss(
                self, data, edge_label_index, edge_label, criterion
            ):
                self.received_edges = edge_label_index.clone()
                self.received_labels = edge_label.clone()
                custom = self.weight.square()
                return custom, {"custom_alignment": custom}

        data = self._training_data()
        negative_edges = torch.tensor([[0, 4], [3, 1]])
        spec = link_prediction.ModelRegistration(
            ContextModel,
            link_prediction.ModelCapabilities(
                uses_message_passing=False,
                uses_conv_layer=False,
                requires_graph_context=True,
            ),
        )
        with patch.dict(
            link_prediction._MODEL_REGISTRY, {"context-test": spec}
        ):
            predictor = link_prediction.LinkPrediction(
                data,
                model_type="context-test",
                hidden_channels=4,
                out_channels=2,
            )
            loss = predictor.train(
                negative_edge_index=negative_edges
            )

        self.assertIs(predictor.model.constructor_data, data)
        self.assertEqual(
            set(predictor.last_loss_components), {"custom_alignment"}
        )
        self.assertTrue(torch.equal(
            predictor.model.received_edges,
            torch.cat([data.edge_label_index, negative_edges], dim=1),
        ))
        self.assertTrue(torch.equal(
            predictor.model.received_labels,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        ))
        self.assertTrue(torch.isfinite(loss))

    def test_symmetric_training_step_returns_finite_loss(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        loss = predictor.train(data)
        self.assertTrue(torch.isfinite(loss))

    def test_asymmetric_test_returns_probabilities_and_sets_eval_mode(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="asymmetric", hidden_channels=4, out_channels=2
        )
        probabilities = predictor.test(data)
        self.assertEqual(probabilities.shape, (data.edge_label_index.size(1),))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertFalse(predictor.model.training)

    def test_symmetric_test_returns_probabilities(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        probabilities = predictor.test(data)
        self.assertEqual(probabilities.shape, (data.edge_label_index.size(1),))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_asymmetric_decoder_rejects_incompatible_feature_dimensions(self):
        model = link_prediction.AGAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.ones((2, 2))
        x_reduced = torch.ones((2, 3))
        with self.assertRaises(RuntimeError):
            model.decode(z, torch.tensor([[0], [1]]), x_reduced)

    def test_trainable_parameter_count_excludes_frozen_parameters(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2, bias=False),
        )
        model[1].weight.requires_grad_(False)

        self.assertEqual(
            link_prediction.count_trainable_parameters(model),
            3 * 4 + 4,
        )
        with self.assertRaises(TypeError):
            link_prediction.count_trainable_parameters(object())


