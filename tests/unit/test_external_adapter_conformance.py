"""Exercise reusable conformance checks against a local adapter file."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from glpbench.models import load_model_adapter_file
import glpbench.models.registry as model_registry
from glpbench.testing import assert_model_adapter_conforms


def _toy_data():
    x = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.8, 0.2],
    ])
    candidates = torch.tensor(
        [[0, 1, 0, 2], [2, 3, 3, 1]], dtype=torch.long
    )
    return Data(
        x=x,
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        ),
        edge_label_index=candidates,
        edge_label=torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )


class TestExternalAdapterConformance(unittest.TestCase):
    def setUp(self):
        self._adapters = dict(model_registry._MODEL_ADAPTERS)
        self._lookup = dict(model_registry._NAME_LOOKUP)
        self._sources = dict(model_registry._MODEL_SOURCES)
        self._loaded_files = dict(model_registry._LOADED_MODEL_FILES)

    def tearDown(self):
        model_registry._MODEL_ADAPTERS.clear()
        model_registry._MODEL_ADAPTERS.update(self._adapters)
        model_registry._NAME_LOOKUP.clear()
        model_registry._NAME_LOOKUP.update(self._lookup)
        model_registry._MODEL_SOURCES.clear()
        model_registry._MODEL_SOURCES.update(self._sources)
        model_registry._LOADED_MODEL_FILES.clear()
        model_registry._LOADED_MODEL_FILES.update(self._loaded_files)

    def test_local_file_adapter_passes_same_conformance_check(self):
        source = '''
from types import MappingProxyType
import torch
from glpbench.contracts import ModelCapabilities, TrainingLoss


class ToyModel(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def logits(self, data, edge_index):
        source, target = edge_index
        return self.scale * (data.x[source] * data.x[target]).sum(dim=1)


class ToyAdapter:
    name = "external-conformance-model"
    capabilities = ModelCapabilities(
        uses_message_passing=False,
        uses_conv_layer=False,
        requires_labels=True,
    )
    parameter_defaults = MappingProxyType({"scale": 1.0})

    def validate_parameters(self, parameters):
        unknown = set(parameters).difference(self.parameter_defaults)
        if unknown:
            raise ValueError(f"unsupported parameters: {sorted(unknown)}")
        scale = parameters.get("scale", 1.0)
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("scale must be positive")
        return MappingProxyType({"scale": float(scale)})

    def build(self, train_data, parameters):
        normalized = self.validate_parameters(parameters)
        return ToyModel(normalized["scale"])

    def compute_training_loss(
        self, model, train_data, edge_label_index, edge_label, criterion
    ):
        total = criterion(model.logits(train_data, edge_label_index), edge_label)
        return TrainingLoss(total, {"reconstruction": total})

    @torch.no_grad()
    def score_edges(self, model, data, edge_index):
        return model.logits(data, edge_index).sigmoid()

    def result_metadata(self, parameters):
        normalized = self.validate_parameters(parameters)
        return MappingProxyType({"scale": normalized["scale"]})


MODEL_ADAPTER = ToyAdapter()
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy_adapter.py"
            path.write_text(source, encoding="utf-8")
            adapter = load_model_adapter_file(path)

            assert_model_adapter_conforms(
                adapter,
                _toy_data(),
                {"scale": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
