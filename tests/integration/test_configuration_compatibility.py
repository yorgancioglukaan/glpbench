import sys
import tempfile
import unittest
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.configuration import load_config  # noqa: E402
from glpbench.exceptions import ConfigurationError  # noqa: E402
from glpbench.runners import audit, benchmark, search  # noqa: E402


class TestConfigurationCompatibility(unittest.TestCase):
    def test_legacy_benchmark_preserves_existing_expansion(self):
        direct = benchmark.load_benchmark_config("configs/example_benchmark.yaml")
        compatible = load_config("configs/example_benchmark.yaml")

        self.assertEqual(compatible, direct)
        self.assertEqual(
            benchmark.expand_benchmark_config(compatible),
            benchmark.expand_benchmark_config(direct),
        )

    def test_legacy_search_preserves_candidates_and_artifacts(self):
        path = "tests/fixtures/pilot_search.yaml"
        direct = search.load_search_config(path)
        compatible = load_config(path)

        self.assertEqual(compatible, direct)
        self.assertEqual(search.generate_candidates(compatible), search.generate_candidates(direct))

    def test_legacy_audit_preserves_benchmark_jobs(self):
        path = "tests/fixtures/convergence_audit.yaml"
        direct = audit.load_audit_config(path)
        compatible = load_config(path)

        self.assertEqual(compatible, direct)
        self.assertEqual(
            audit.expand_audit_config(compatible),
            audit.expand_audit_config(direct),
        )

    def test_version_two_selected_benchmark_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected.yaml"
            selected.write_text(yaml.safe_dump({
                "version": 1,
                "kind": "selected-hyperparameters",
                "selections": {
                    "cora": {
                        "gae": {
                            "hyperparameters": {
                                "hidden_channels": 16,
                                "out_channels": 8,
                                "learning_rate": 0.01,
                                "weight_decay": 0.0,
                                "normalize_features": True,
                                "conv_layer_type": "GCNConv",
                            }
                        }
                    }
                },
            }), encoding="utf-8")
            path = root / "benchmark.yaml"
            path.write_text(yaml.safe_dump({
                "version": 2,
                "kind": "selected-hyperparameter-benchmark",
                "output": str(root / "results.csv"),
                "seeds": [0],
                "selected_configurations": {
                    "path": str(selected),
                    "granularity": ["dataset", "architecture"],
                    "require_all_pairs": True,
                },
                "datasets": ["cora"],
                "models": ["gae"],
            }), encoding="utf-8")

            direct = benchmark.load_benchmark_config(path)
            compatible = load_config(path)

            self.assertEqual(compatible, direct)
            self.assertEqual(
                benchmark.expand_benchmark_config(compatible),
                benchmark.expand_benchmark_config(direct),
            )

    def test_training_policy_ablation_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ablation.yaml"
            path.write_text(yaml.safe_dump({
                "version": 1,
                "kind": "training-policy-ablation",
                "output": str(root / "results.csv"),
                "error_output": str(root / "errors.jsonl"),
                "epoch_history_output": str(root / "epochs.csv"),
                "data_root": "data",
                "seeds": [0],
                "selected_configurations": {
                    "path": str(root / "selected.yaml"),
                    "granularity": ["dataset", "architecture"],
                    "require_all_pairs": True,
                },
                "datasets": ["cora"],
                "models": ["gae"],
                "splits": {
                    "protocol": "matched-semi-inductive",
                    "query_fraction": 0.05,
                    "validation_query_fraction": 0.05,
                    "train_ratio": 0.8,
                    "paired_across_policies": True,
                },
                "evaluation": {
                    "split_types": ["transductive", "inductive"],
                    "primary_split": "inductive",
                    "primary_metric": "roc_auc",
                    "secondary_metrics": ["average_precision"],
                },
                "training": {
                    "device": "cpu",
                    "selection_metric": "roc_auc",
                    "log_epoch_history": True,
                    "epoch_history_metrics": [
                        "training_loss", "validation_roc_auc"
                    ],
                },
                "training_policies": [{
                    "name": "fixed",
                    "max_epochs": 10,
                    "checkpoint": "final",
                    "early_stopping": {"enabled": False},
                }],
            }), encoding="utf-8")

            self.assertEqual(
                load_config(path), benchmark.load_benchmark_config(path)
            )

    def test_selected_hyperparameters_remain_loadable(self):
        selected = load_config("tests/fixtures/audit_selected_configurations.yaml")
        self.assertIn("cora", selected)
        self.assertIn("gae", selected["cora"])

    def test_mixed_or_missing_version_markers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump({
                    "version": 1,
                    "schema_version": 1,
                    "kind": "benchmark",
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "cannot define both"):
                load_config(path)

            path.write_text(yaml.safe_dump({"kind": "benchmark"}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "must define schema_version"):
                load_config(path)

    def test_typed_canonical_loaders_do_not_accept_legacy_documents(self):
        from glpbench.configuration import load_benchmark_config

        with self.assertRaisesRegex(ConfigurationError, "legacy format"):
            load_benchmark_config("configs/example_benchmark.yaml")


if __name__ == "__main__":
    unittest.main()
