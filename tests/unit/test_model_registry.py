"""Tests for public model-adapter registration and lookup."""

import unittest
from pathlib import Path
import tempfile

import torch

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.exceptions import CompatibilityError
from glpbench.interfaces import ModelAdapter
from glpbench.models import (
    available_model_types,
    get_model_adapter,
    get_model_adapter_source,
    load_model_adapter_file,
    normalize_model_name,
    register_model_adapter,
    resolve_model_reference,
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
        self._sources = dict(registry._MODEL_SOURCES)
        self._loaded_files = dict(registry._LOADED_MODEL_FILES)

    def tearDown(self):
        registry._MODEL_ADAPTERS.clear()
        registry._MODEL_ADAPTERS.update(self._adapters)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)
        registry._MODEL_SOURCES.clear()
        registry._MODEL_SOURCES.update(self._sources)
        registry._LOADED_MODEL_FILES.clear()
        registry._LOADED_MODEL_FILES.update(self._loaded_files)

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

    def test_loads_one_adapter_from_local_file_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model3.py"
            path.write_text(
                "from tests.unit.test_model_registry import _ToyAdapter\n"
                "MODEL_ADAPTER = _ToyAdapter()\n",
                encoding="utf-8",
            )

            adapter = load_model_adapter_file(path, reference="./model3.py")
            repeated = load_model_adapter_file(path)
            source = get_model_adapter_source("toy-model")

        self.assertIs(adapter, repeated)
        self.assertIs(get_model_adapter("toy-model"), adapter)
        self.assertEqual(source.reference, "./model3.py")
        self.assertEqual(source.path, path.resolve())
        self.assertEqual(len(source.sha256), 64)

    def test_resolves_relative_model_file_from_configuration_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = root / "models"
            model_directory.mkdir()
            path = model_directory / "model3.py"
            path.write_text(
                "from tests.unit.test_model_registry import _ToyAdapter\n"
                "MODEL_ADAPTER = _ToyAdapter()\n",
                encoding="utf-8",
            )

            name = resolve_model_reference(
                "models/model3.py", base_dir=root
            )

        self.assertEqual(name, "toy-model")

    def test_local_file_failures_leave_registry_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            before = available_model_types()

            with self.assertRaisesRegex(CompatibilityError, "MODEL_ADAPTER"):
                load_model_adapter_file(path)

        self.assertEqual(available_model_types(), before)
        self.assertIsNone(get_model_adapter_source("symmetric"))

    def test_local_file_rejects_invalid_paths_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(CompatibilityError, "does not exist"):
                load_model_adapter_file(root / "missing.py")
            text = (
                "from tests.unit.test_model_registry import _ToyAdapter\n"
                "MODEL_ADAPTER = _ToyAdapter()\n"
            )
            first = root / "first.py"
            second = root / "second.py"
            first.write_text(text, encoding="utf-8")
            second.write_text(text, encoding="utf-8")
            load_model_adapter_file(first)
            with self.assertRaisesRegex(ValueError, "already registered"):
                load_model_adapter_file(second)

    def test_changed_local_file_is_not_silently_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model3.py"
            path.write_text(
                "from tests.unit.test_model_registry import _ToyAdapter\n"
                "MODEL_ADAPTER = _ToyAdapter()\n",
                encoding="utf-8",
            )
            load_model_adapter_file(path)
            path.write_text(path.read_text() + "# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(CompatibilityError, "changed"):
                load_model_adapter_file(path)

if __name__ == "__main__":
    unittest.main()
