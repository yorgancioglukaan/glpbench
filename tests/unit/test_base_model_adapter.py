"""Contract tests for the public model-adapter base class."""

from types import MappingProxyType
import unittest

import torch
from torch_geometric.data import Data

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.exceptions import ConfigurationError
from glpbench.interfaces import ModelAdapter
from glpbench.models import BaseModelAdapter, GAEAdapter, get_model_adapter
from glpbench.testing import assert_model_adapter_conforms


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, x, edge_index):
        return x * self.weight

    def decode_edges(self, representations, edge_index, data):
        source, target = edge_index
        return (representations[source] * representations[target]).sum(dim=1)

    def compute_training_loss(
        self, train_data, edge_label_index, edge_label, criterion
    ):
        scores = self.decode_edges(
            self.encode(train_data.x, train_data.edge_index),
            edge_label_index,
            train_data,
        )
        loss = criterion(scores, edge_label)
        return loss, {"reconstruction": loss}


class _ToyAdapter(BaseModelAdapter):
    name = "base-toy"
    capabilities = ModelCapabilities(requires_labels=True)
    defaults = MappingProxyType({"scale": 1.0, "options": {"bias": 0.0}})

    def _validate_values(self, parameters):
        if parameters["scale"] <= 0:
            raise ValueError("scale must be positive")

    def build(self, train_data, parameters):
        self.validate_parameters(parameters)
        return _ToyModel()


class TestBaseModelAdapter(unittest.TestCase):
    def test_abstract_construction_and_required_declarations(self):
        with self.assertRaises(TypeError):
            BaseModelAdapter()

        for declaration in ("name", "capabilities", "defaults"):
            attributes = {
                "name": "missing-field",
                "capabilities": ModelCapabilities(),
                "defaults": MappingProxyType({}),
                "build": lambda self, data, parameters: _ToyModel(),
            }
            del attributes[declaration]
            adapter_type = type("IncompleteAdapter", (BaseModelAdapter,), attributes)
            with self.subTest(declaration=declaration), self.assertRaises(TypeError):
                adapter_type()

    def test_defaults_are_immutable_and_input_is_not_mutated(self):
        adapter = _ToyAdapter()
        supplied = {"scale": 2.0}
        normalized = adapter.validate_parameters(supplied)
        self.assertEqual(supplied, {"scale": 2.0})
        self.assertEqual(normalized["scale"], 2.0)
        self.assertEqual(normalized["options"]["bias"], 0.0)
        with self.assertRaises(TypeError):
            adapter.parameter_defaults["scale"] = 3.0
        with self.assertRaises(TypeError):
            normalized["options"]["bias"] = 1.0
        with self.assertRaises(ConfigurationError):
            adapter.validate_parameters({"unknown": 1})
        with self.assertRaises(ValueError):
            adapter.validate_parameters({"scale": 0})

    def test_standard_loss_scoring_and_empty_metadata(self):
        adapter = _ToyAdapter()
        data = Data(
            x=torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        candidates = torch.tensor([[0, 1], [1, 2]])
        labels = torch.tensor([1.0, 0.0])
        model = adapter.build(data, {})
        loss = adapter.compute_training_loss(
            model, data, candidates, labels, torch.nn.BCEWithLogitsLoss()
        )
        self.assertIsInstance(loss, TrainingLoss)
        loss.total.backward()
        self.assertIsNotNone(model.weight.grad)
        scores = adapter.score_edges(model, data, candidates)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))
        self.assertEqual(dict(adapter.result_metadata({})), {})
        self.assertIsInstance(adapter, ModelAdapter)

    def test_bundled_adapters_use_base_and_structural_registry_remains(self):
        self.assertIsInstance(GAEAdapter(), BaseModelAdapter)
        self.assertIsInstance(get_model_adapter("gae"), ModelAdapter)

    def test_subclass_passes_reusable_conformance_suite(self):
        data = Data(
            x=torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_label_index=torch.tensor([[0, 1], [1, 2]]),
            edge_label=torch.tensor([1.0, 0.0]),
        )
        assert_model_adapter_conforms(_ToyAdapter(), data, {"scale": 2.0})


if __name__ == "__main__":
    unittest.main()
