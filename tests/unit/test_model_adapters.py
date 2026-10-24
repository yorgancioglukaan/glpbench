"""Tests for the bundled model adapters."""

from types import MappingProxyType
import unittest

import torch
from torch_geometric.data import Data

from glpbench.exceptions import ConfigurationError, TrainingError
from glpbench.interfaces import ModelAdapter
from glpbench.models import (
    AGAEAdapter,
    AGAE_Model,
    AGAEMLPAdapter,
    DEALAdapter,
    DEALModel,
    FeatureCosineAdapter,
    FeatureCosineModel,
    FeatureMLPAdapter,
    FeatureMLPModel,
    GAEAdapter,
    GAE_Model,
    GAEMLPAdapter,
    LinkPrediction,
    MLP_AGAE_Model,
    MLP_GAE_Model,
    VGAEAdapter,
    VGAE_Model,
    get_model_adapter,
    resolve_model_parameters,
)
from glpbench.testing import assert_model_adapter_conforms


ADAPTER_CASES = (
    (GAEAdapter, "gae", GAE_Model),
    (GAEMLPAdapter, "gae-mlp", MLP_GAE_Model),
    (AGAEAdapter, "agae", AGAE_Model),
    (AGAEMLPAdapter, "agae-mlp", MLP_AGAE_Model),
    (VGAEAdapter, "vgae", VGAE_Model),
    (DEALAdapter, "deal", DEALModel),
    (FeatureCosineAdapter, "feature-cosine", FeatureCosineModel),
    (FeatureMLPAdapter, "feature-mlp", FeatureMLPModel),
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
    edge_label_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index,
        edge_label=torch.ones(edge_label_index.size(1)),
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
    common = {
        "hidden_channels": 5,
        "out_channels": 3,
        "conv_layer_type": "GCNConv",
        "gatv2_kwargs": {},
    }
    if name == "gae-mlp":
        return {**common, "decoder_hidden_channels": 4, "decoder_dropout": 0.1}
    if name == "agae":
        return {
            **common,
            "regularization_type": "cosine-alignment",
            "regularization_strength": 0.2,
        }
    if name == "agae-mlp":
        return {
            **common,
            "decoder_hidden_channels": 4,
            "decoder_dropout": 0.1,
            "regularization_type": "cosine-alignment",
            "regularization_strength": 0.2,
        }
    if name == "vgae":
        return {**common, "kl_weight": 0.3}
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
    return common


class TestBundledModelAdapters(unittest.TestCase):
    """Verify adapter conformance without changing runtime registration."""

    def setUp(self):
        self.data = _toy_data()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.loss_edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long
        )
        self.loss_edge_label = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    def test_parameter_resolution_selects_only_adapter_declarations(self):
        supplied = {
            "hidden_channels": 7,
            "out_channels": 3,
            "conv_layer_type": "GCNConv",
            "gatv2_kwargs": {},
            "kl_weight": 0.4,
            "decoder_dropout": 0.2,
            "theta_ss": 0.5,
        }

        graph = resolve_model_parameters(GAEAdapter(), supplied)
        vgae = resolve_model_parameters(VGAEAdapter(), supplied)
        feature = resolve_model_parameters(FeatureMLPAdapter(), supplied)

        self.assertNotIn("kl_weight", graph)
        self.assertEqual(vgae["kl_weight"], 0.4)
        self.assertEqual(feature["decoder_dropout"], 0.2)
        self.assertNotIn("hidden_channels", feature)

    def test_link_prediction_accepts_explicit_adapter_parameters(self):
        parameters = resolve_model_parameters(
            GAEAdapter(), _parameters("gae")
        )
        predictor = LinkPrediction(
            self.data,
            model_type="gae",
            model_parameters=parameters,
            learning_rate=0.01,
            weight_decay=0.0,
        )

        self.assertEqual(dict(predictor.model_parameters), dict(parameters))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            LinkPrediction(
                self.data,
                model_type="gae",
                model_parameters=parameters,
                hidden_channels=8,
            )

    def test_all_adapters_satisfy_contract_and_match_capabilities(self):
        for adapter_class, name, _ in ADAPTER_CASES:
            with self.subTest(name=name):
                adapter = adapter_class()
                self.assertIsInstance(adapter, ModelAdapter)
                self.assertEqual(adapter.name, name)
                self.assertEqual(
                    adapter.capabilities,
                    get_model_adapter(name).capabilities,
                )

    def test_all_bundled_adapters_pass_reusable_conformance_checks(self):
        for adapter_class, name, _ in ADAPTER_CASES:
            with self.subTest(name=name):
                assert_model_adapter_conforms(
                    adapter_class(),
                    self.data,
                    _parameters(name),
                    edge_label_index=self.loss_edge_index,
                    edge_label=self.loss_edge_label,
                    criterion=self.criterion,
                )

    def test_parameter_validation_is_immutable_and_rejects_unknown_fields(self):
        for adapter_class, name, _ in ADAPTER_CASES:
            with self.subTest(name=name):
                adapter = adapter_class()
                normalized = adapter.validate_parameters(_parameters(name))
                self.assertIsInstance(normalized, MappingProxyType)
                with self.assertRaises(TypeError):
                    normalized["new"] = 1
                with self.assertRaises(ConfigurationError):
                    adapter.validate_parameters({"not_supported": True})

        graph_parameters = GAEAdapter().validate_parameters({
            "gatv2_kwargs": {"heads": 2}
        })
        self.assertIsInstance(graph_parameters["gatv2_kwargs"], MappingProxyType)
        with self.assertRaises(TypeError):
            graph_parameters["gatv2_kwargs"]["heads"] = 4

    def test_build_matches_existing_facade_construction(self):
        for adapter_class, name, model_class in ADAPTER_CASES:
            with self.subTest(name=name):
                parameters = _parameters(name)
                torch.manual_seed(311)
                facade = LinkPrediction(
                    self.data,
                    model_type=name,
                    learning_rate=0.01,
                    weight_decay=0.0,
                    **parameters,
                )
                torch.manual_seed(311)
                adapted = adapter_class().build(self.data, parameters)
                self.assertIsInstance(adapted, model_class)
                self.assertEqual(
                    tuple(facade.model.state_dict()), tuple(adapted.state_dict())
                )
                for key, value in facade.model.state_dict().items():
                    torch.testing.assert_close(value, adapted.state_dict()[key])

    def test_feature_cosine_rejects_training_loss(self):
        adapter = FeatureCosineAdapter()
        model = adapter.build(self.data, {})
        with self.assertRaises(TrainingError):
            adapter.compute_training_loss(
                model,
                self.data,
                self.loss_edge_index,
                self.loss_edge_label,
                self.criterion,
            )

    def test_metadata_contains_only_architecture_specific_fields(self):
        self.assertEqual(dict(GAEAdapter().result_metadata({})), {})
        self.assertEqual(
            dict(VGAEAdapter().result_metadata({"kl_weight": 0.4})),
            {"kl_weight": 0.4},
        )
        agae_metadata = AGAEMLPAdapter().result_metadata({
            "regularization_type": "latent-l2",
            "regularization_strength": 0.1,
            "decoder_hidden_channels": 12,
        })
        self.assertEqual(agae_metadata["regularization_type"], "latent-l2")
        self.assertEqual(agae_metadata["regularization_strength"], 0.1)
        self.assertEqual(agae_metadata["decoder_hidden_channels"], 12)
        self.assertEqual(agae_metadata["decoder_dropout"], 0.0)
        deal_metadata = DEALAdapter().result_metadata({})
        self.assertEqual(deal_metadata["deal_theta_ss"], 1.0)
        self.assertEqual(deal_metadata["deal_similarity_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
