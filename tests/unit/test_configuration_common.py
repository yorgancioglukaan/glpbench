import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.configuration.common import (  # noqa: E402
    parse_dataset_config,
    parse_execution_config,
    parse_model_config,
    parse_split_config,
    parse_training_config,
)
from glpbench.exceptions import ConfigurationError  # noqa: E402


class TestCommonConfigurationSections(unittest.TestCase):
    def test_sections_are_normalized_and_frozen(self):
        datasets = parse_dataset_config({"names": ["cora"], "root": "graphs"})
        execution = parse_execution_config({"seeds": [2, 4]})
        model = parse_model_config(
            {"architecture": "symmetric", "parameters": {"dropout": 0.1}},
            ("models", 0),
        )

        self.assertEqual(datasets.root, Path("graphs"))
        self.assertEqual(execution.seeds, (2, 4))
        with self.assertRaises(FrozenInstanceError):
            datasets.root = Path("other")
        with self.assertRaises(TypeError):
            model.parameters["dropout"] = 0.2

    def test_split_requires_exactly_one_test_query_measure(self):
        with self.assertRaisesRegex(ConfigurationError, "exactly one"):
            parse_split_config({
                "protocol": "matched", "train_ratio": 0.8,
                "query_size": 10, "query_fraction": 0.1,
            })
        with self.assertRaisesRegex(ConfigurationError, "exactly one"):
            parse_split_config({"protocol": "matched", "train_ratio": 0.8})

    def test_training_reports_deep_paths_and_validates_epoch_relationship(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            r"training\.early_stopping\.patience is required",
        ):
            parse_training_config({
                "max_epochs": 20,
                "early_stopping": {"enabled": True},
            })
        with self.assertRaisesRegex(
            ConfigurationError,
            r"training\.early_stopping\.minimum_epochs cannot exceed",
        ):
            parse_training_config({
                "max_epochs": 20,
                "early_stopping": {
                    "enabled": True, "patience": 2, "minimum_epochs": 21,
                },
            })

    def test_every_nested_section_rejects_unknown_keys(self):
        with self.assertRaisesRegex(
            ConfigurationError, r"datasets\.normalise is not a recognized field"
        ):
            parse_dataset_config({"names": ["cora"], "normalise": True})
        with self.assertRaisesRegex(
            ConfigurationError,
            r"training\.scheduler\.factor is not a recognized field",
        ):
            parse_training_config({
                "max_epochs": 10,
                "scheduler": {"type": "step", "factor": 0.5},
            })


if __name__ == "__main__":
    unittest.main()
