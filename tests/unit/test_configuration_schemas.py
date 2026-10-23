import sys
import tempfile
import unittest
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.configuration.ablation import parse_ablation_config  # noqa: E402
from glpbench.configuration.audit import parse_audit_config  # noqa: E402
from glpbench.configuration.benchmark import parse_benchmark_config  # noqa: E402
from glpbench.configuration.search import parse_search_config  # noqa: E402
from glpbench.configuration.loading import (  # noqa: E402
    load_benchmark_config,
    load_config,
)
from glpbench.configuration.validation import validate_config  # noqa: E402
from glpbench.configuration.resolution import resolve_experiment_specs  # noqa: E402
from glpbench.exceptions import ConfigurationError  # noqa: E402
import glpbench.models.registry as model_registry  # noqa: E402


def _common(kind):
    return {
        "schema_version": 1,
        "kind": kind,
        "datasets": {"names": ["cora"], "root": "data"},
        "split": {
            "protocol": "matched-semi-inductive",
            "train_ratio": 0.8,
            "query_size": 10,
            "validation_query_size": 10,
        },
        "models": [{"architecture": "symmetric"}],
        "training": {"max_epochs": 20},
        "execution": {"seeds": [0], "device": "cpu"},
        "output": {"results": "out/results.csv"},
    }


class TestCanonicalConfigurationSchemas(unittest.TestCase):
    def test_benchmark_parses_minimal_canonical_document(self):
        config = parse_benchmark_config(_common("benchmark"))
        self.assertEqual(config.datasets.names, ("cora",))
        self.assertEqual(config.models[0].architecture, "symmetric")
        self.assertEqual(config.output.results, Path("out/results.csv"))

    def test_search_parses_spaces_and_freezes_them(self):
        document = _common("hyperparameter-search")
        document["models"] = [{
            "architecture": "symmetric",
            "spaces": {
                "hidden_channels": {
                    "values": [32, 64], "sampling": "random"
                }
            },
        }]
        document["search"] = {
            "strategy": "seeded-random", "num_candidates": 3,
            "candidate_seed": 9, "include_default": True,
        }
        document["selection"] = {
            "metric": "roc_auc", "aggregation": "mean",
            "validation_setting": "semi-inductive",
        }
        document["output"] = {
            "history": "out/history.csv",
            "selected_configurations": "out/selected.yaml",
        }
        config = parse_search_config(document)
        self.assertEqual(
            config.models[0].spaces["hidden_channels"].values, (32, 64)
        )
        with self.assertRaises(TypeError):
            config.models[0].spaces["other"] = None

    def test_ablation_parses_named_policies(self):
        document = _common("training-policy-ablation")
        del document["training"]
        document["policies"] = [
            {"name": "fixed", "training": {"max_epochs": 20}},
            {"name": "long", "training": {"max_epochs": 50}},
        ]
        config = parse_ablation_config(document)
        self.assertEqual(
            tuple(item.name for item in config.policies), ("fixed", "long")
        )

    def test_audit_parses_optional_selected_configuration_path(self):
        document = _common("validation-convergence-audit")
        document["selected_configurations"] = "out/selected.yaml"
        config = parse_audit_config(document)
        self.assertEqual(config.selected_configurations, Path("out/selected.yaml"))

    def test_roots_reject_unknown_fields_and_wrong_versions(self):
        unknown = _common("benchmark")
        unknown["typo"] = True
        with self.assertRaisesRegex(ConfigurationError, "typo is not a recognized field"):
            parse_benchmark_config(unknown)
        version = _common("benchmark")
        version["schema_version"] = 2
        with self.assertRaisesRegex(ConfigurationError, "schema_version must be 1"):
            parse_benchmark_config(version)

    def test_semantic_validation_reports_registered_name_paths(self):
        document = _common("benchmark")
        document["datasets"]["names"] = ["not-a-dataset"]
        with self.assertRaisesRegex(
            ConfigurationError, r"datasets\.names\[0\] references unknown"
        ):
            validate_config(parse_benchmark_config(document))

        document = _common("benchmark")
        document["models"] = [{"architecture": "feature-cosine", "conv_layer_types": ["GCNConv"]}]
        with self.assertRaisesRegex(
            ConfigurationError, r"models\[0\]\.conv_layer_types is not supported"
        ):
            validate_config(parse_benchmark_config(document))

    def test_load_config_dispatches_and_typed_loader_checks_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(_common("benchmark")), encoding="utf-8"
            )
            config = load_config(path)
            self.assertEqual(config.kind, "benchmark")
            self.assertEqual(load_benchmark_config(path), config)

            wrong = _common("validation-convergence-audit")
            path.write_text(yaml.safe_dump(wrong), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "kind must be 'benchmark'"):
                load_benchmark_config(path)

    def test_load_config_resolves_local_model_relative_to_config(self):
        adapters = dict(model_registry._MODEL_ADAPTERS)
        lookup = dict(model_registry._NAME_LOOKUP)
        sources = dict(model_registry._MODEL_SOURCES)
        loaded_files = dict(model_registry._LOADED_MODEL_FILES)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                models = root / "models"
                models.mkdir()
                adapter_path = models / "local_model.py"
                adapter_path.write_text(
                    "from glpbench.models import FeatureCosineAdapter\n"
                    "class LocalAdapter(FeatureCosineAdapter):\n"
                    "    name = 'local-config-model'\n"
                    "MODEL_ADAPTER = LocalAdapter()\n",
                    encoding="utf-8",
                )
                document = _common("benchmark")
                document["models"] = [{"architecture": "models/local_model.py"}]
                config_path = root / "benchmark.yaml"
                config_path.write_text(
                    yaml.safe_dump(document), encoding="utf-8"
                )

                config = load_config(config_path)

                self.assertEqual(
                    config.models[0].architecture, "local-config-model"
                )
                source = model_registry.get_model_adapter_source(
                    "local-config-model"
                )
                self.assertEqual(source.path, adapter_path.resolve())
                self.assertEqual(source.reference, "models/local_model.py")
        finally:
            model_registry._MODEL_ADAPTERS.clear()
            model_registry._MODEL_ADAPTERS.update(adapters)
            model_registry._NAME_LOOKUP.clear()
            model_registry._NAME_LOOKUP.update(lookup)
            model_registry._MODEL_SOURCES.clear()
            model_registry._MODEL_SOURCES.update(sources)
            model_registry._LOADED_MODEL_FILES.clear()
            model_registry._LOADED_MODEL_FILES.update(loaded_files)

    def test_benchmark_resolves_to_runtime_experiment_contracts(self):
        document = _common("benchmark")
        document["datasets"]["names"] = ["cora", "citeseer"]
        document["execution"]["seeds"] = [0, 1]
        document["models"] = [{
            "architecture": "asymmetric",
            "conv_layer_types": ["GCNConv", "SAGEConv"],
            "reduction_methods": ["pca"],
            "parameters": {"hidden_channels": 64, "out_channels": 32},
        }]

        specs = resolve_experiment_specs(parse_benchmark_config(document))

        self.assertEqual(len(specs), 8)
        self.assertEqual({spec.dataset for spec in specs}, {"cora", "citeseer"})
        self.assertEqual({spec.seed for spec in specs}, {0, 1})
        self.assertEqual(
            {spec.model_parameters["conv_layer_type"] for spec in specs},
            {"GCNConv", "SAGEConv"},
        )
        self.assertTrue(all(spec.preprocessing == "pca" for spec in specs))


if __name__ == "__main__":
    unittest.main()
