import sys
import tempfile
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.configuration.parsing import (  # noqa: E402
    load_yaml_mapping,
    reject_unknown_keys,
    render_path,
    require_integer,
    require_mapping,
    require_unique,
)
from glpbench.exceptions import ConfigurationError  # noqa: E402


class TestConfigurationParsing(unittest.TestCase):
    def test_paths_render_nested_fields_and_indexes(self):
        self.assertEqual(
            render_path(("models", 2, "parameters", "dropout")),
            "models[2].parameters.dropout",
        )
        self.assertEqual(render_path(()), "<document>")

    def test_unknown_and_missing_types_report_full_paths(self):
        with self.assertRaisesRegex(
            ConfigurationError, r"training\.unknown is not a recognized field"
        ):
            reject_unknown_keys(
                {"epochs": 10, "unknown": True}, {"epochs"}, ("training",)
            )
        with self.assertRaisesRegex(
            ConfigurationError, r"training\.epochs must be an integer"
        ):
            require_integer(True, ("training", "epochs"))
        with self.assertRaisesRegex(
            ConfigurationError, r"models\[1\] must be a mapping"
        ):
            require_mapping("bad", ("models", 1))

    def test_unique_sequence_identifies_duplicate_index(self):
        with self.assertRaisesRegex(
            ConfigurationError, r"execution\.seeds\[2\] duplicates"
        ):
            require_unique([1, 2, 1], ("execution", "seeds"))

    def test_yaml_loader_requires_mapping_and_wraps_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sequence.yaml"
            sequence.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "must be a mapping"):
                load_yaml_mapping(sequence)

            invalid = root / "invalid.yaml"
            invalid.write_text("field: [\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "invalid YAML"):
                load_yaml_mapping(invalid)


if __name__ == "__main__":
    unittest.main()
