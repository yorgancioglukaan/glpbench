"""Tests for public model-adapter registration and lookup."""

import unittest

import torch

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.exceptions import CompatibilityError
from glpbench.interfaces import ModelAdapter
from glpbench.models import (
    available_model_types,
    get_model_adapter,
    normalize_model_name,
    register_model_adapter,
)
import glpbench.models.registry as registry


class _ToyAdapter:
    name = "toy-model"
    capabilities = ModelCapabilities(
        uses_message_passing=False,
        uses_conv_layer=False,
    )

    def __init__(self, marker=0):
        self.marker = marker

    def validate_parameters(self, parameters):
        return dict(parameters)

    def build(self, train_data, parameters):
        return torch.nn.Identity()

    def compute_training_loss(
        self, model, train_data, edge_label_index, edge_label, criterion
    ):
        loss = torch.zeros((), requires_grad=True)
        return TrainingLoss(loss, {"toy": loss})

    def score_edges(self, model, data, edge_index):
        return torch.zeros(edge_index.size(1))

    def result_metadata(self, parameters):
        return {"marker": self.marker}


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self._adapters = dict(registry._MODEL_ADAPTERS)
        self._lookup = dict(registry._NAME_LOOKUP)

    def tearDown(self):
        registry._MODEL_ADAPTERS.clear()
        registry._MODEL_ADAPTERS.update(self._adapters)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)

    def test_builtins_are_registered_and_conformant(self):
        self.assertEqual(available_model_types(), (
            "asymmetric",
            "asymmetric-mlp",
            "deal",
            "feature-cosine",
            "feature-mlp",
            "mlp",
            "symmetric",
            "variational",
        ))
        for name in available_model_types():
            with self.subTest(name=name):
                adapter = get_model_adapter(name)
                self.assertIsInstance(adapter, ModelAdapter)
                self.assertEqual(adapter.name, name)

    def test_names_are_case_and_separator_insensitive(self):
        self.assertEqual(normalize_model_name(" Feature_MLP "), "featuremlp")
        expected = get_model_adapter("feature-mlp")
        self.assertIs(get_model_adapter("FEATURE MLP"), expected)
        self.assertIs(get_model_adapter("feature_mlp"), expected)

    def test_registration_supports_aliases_and_explicit_replacement(self):
        first = register_model_adapter(
            _ToyAdapter(marker=1), aliases=("Toy Adapter",)
        )
        self.assertIs(get_model_adapter("toy_adapter"), first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model_adapter(_ToyAdapter(marker=2))

        second = register_model_adapter(
            _ToyAdapter(marker=2), aliases=("replacement",), replace=True
        )
        self.assertIs(get_model_adapter("replacement"), second)
        with self.assertRaises(CompatibilityError):
            get_model_adapter("Toy Adapter")

    def test_alias_collisions_are_rejected(self):
        register_model_adapter(_ToyAdapter(), aliases=("reserved",))

        class OtherAdapter(_ToyAdapter):
            name = "other-model"

        with self.assertRaisesRegex(ValueError, "collides"):
            register_model_adapter(OtherAdapter(), aliases=("reserved",))

    def test_invalid_adapters_and_registry_arguments_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "ModelAdapter"):
            register_model_adapter(object())

        class InvalidCapabilities(_ToyAdapter):
            name = "invalid-capabilities"
            capabilities = object()

        with self.assertRaisesRegex(TypeError, "ModelCapabilities"):
            register_model_adapter(InvalidCapabilities())
        with self.assertRaisesRegex(TypeError, "aliases"):
            register_model_adapter(_ToyAdapter(), aliases="alias")
        with self.assertRaisesRegex(TypeError, "replace"):
            register_model_adapter(_ToyAdapter(), replace=1)
        with self.assertRaises(ValueError):
            normalize_model_name("---")

    def test_unknown_model_error_lists_available_models(self):
        with self.assertRaisesRegex(
            CompatibilityError, "Available models:.*symmetric"
        ):
            get_model_adapter("missing")

if __name__ == "__main__":
    unittest.main()
