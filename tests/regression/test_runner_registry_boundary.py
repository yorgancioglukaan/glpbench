"""Prevent concrete data-preparation dependencies from returning to runners."""

import ast
import unittest
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_RUNNERS = _PROJECT_ROOT / "src" / "glpbench" / "runners"
_BANNED = {
    "generate_comparable_splits",
    "generate_validation_splits",
    "generate_transductive_splits",
    "inductive_split",
    "select_query_node_sets",
    "fit_reducer",
    "normalize_graph_features",
    "available_reduction_methods",
}


class TestRunnerRegistryBoundary(unittest.TestCase):
    def test_runners_do_not_import_concrete_preparation_implementations(self):
        violations = []
        for path in sorted(_RUNNERS.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for imported in node.names:
                        if imported.name in _BANNED:
                            violations.append(f"{path.name}:{node.lineno}:{imported.name}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
