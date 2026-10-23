import sys
import unittest
from pathlib import Path

import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.contracts import (  # noqa: E402
    AuditReport,
    CandidateOrientation,
    EndpointMode,
    ModelCapabilities,
    ProtocolRequirements,
)
from glpbench.exceptions import (  # noqa: E402
    CompatibilityError,
    ConfigurationError,
    DatasetError,
    GLPBenchError,
    PersistenceError,
    ProtocolError,
    TrainingError,
)
from glpbench.interfaces import (  # noqa: E402
    DatasetProvider,
    EvaluationMetric,
    FeaturePreprocessor,
    LinkPredictionModel,
    ResultStore,
    SplitProtocol,
)
from glpbench.logging import CSVResultLogger  # noqa: E402


class _DatasetProvider:
    name = "toy"

    def load(self, root, device):
        return None


class _SplitProtocol:
    name = "toy-split"
    requirements = ProtocolRequirements(
        endpoint_modes=(EndpointMode.VISIBLE_VISIBLE,),
        candidate_orientations=(CandidateOrientation.UNDIRECTED,),
    )

    def split(self, data, seed):
        return None

    def audit(self, data, splits):
        return AuditReport(passed=True, checks={"disjoint": True})


class _FeaturePreprocessor:
    name = "identity"
    requires_fitting = False
    supports_sparse_inputs = True
    uses_randomness = False

    def fit(self, x, fit_mask, seed=None):
        return self

    def transform(self, x):
        return x


class _LinkPredictionModel:
    capabilities = ModelCapabilities()

    def fit(self, train_data, validation_data, training_policy):
        return None

    def score_edges(self, data, edge_index):
        return torch.empty(edge_index.size(1))


class _Metric:
    name = "mean-score"
    input_kind = "raw-scores"

    def compute(self, labels, scores):
        return float(scores.mean())


class _ResultStore:
    def write(self, specification, training, evaluations):
        return None


class TestComponentInterfaces(unittest.TestCase):
    def test_complete_structural_implementations_satisfy_protocols(self):
        implementations = (
            (_DatasetProvider(), DatasetProvider),
            (_SplitProtocol(), SplitProtocol),
            (_FeaturePreprocessor(), FeaturePreprocessor),
            (_LinkPredictionModel(), LinkPredictionModel),
            (_Metric(), EvaluationMetric),
            (_ResultStore(), ResultStore),
        )
        for implementation, protocol in implementations:
            with self.subTest(protocol=protocol.__name__):
                self.assertIsInstance(implementation, protocol)

    def test_missing_required_member_fails_runtime_check(self):
        class IncompleteMetric:
            name = "incomplete"

            def compute(self, labels, scores):
                return 0.0

        self.assertNotIsInstance(IncompleteMetric(), EvaluationMetric)

    def test_csv_logger_implements_result_store_boundary(self):
        self.assertIsInstance(CSVResultLogger("unused.csv"), ResultStore)


class TestAuditReport(unittest.TestCase):
    def test_report_freezes_checks_and_details(self):
        report = AuditReport(
            passed=True,
            checks={"positive-exclusion": True},
            details={"positive-exclusion": "No overlap found."},
        )
        with self.assertRaises(TypeError):
            report.checks["query-disjointness"] = True

    def test_report_rejects_inconsistent_summary(self):
        with self.assertRaisesRegex(ValueError, "must agree"):
            AuditReport(passed=True, checks={"disjoint": False})


class TestExceptionContracts(unittest.TestCase):
    def test_component_errors_share_one_base_type(self):
        for error_type in (
            ConfigurationError,
            DatasetError,
            ProtocolError,
            CompatibilityError,
            TrainingError,
            PersistenceError,
        ):
            with self.subTest(error_type=error_type.__name__):
                self.assertTrue(issubclass(error_type, GLPBenchError))

    def test_error_preserves_machine_readable_context(self):
        error = ProtocolError(
            "split audit failed",
            component="semi-inductive",
            dataset="cora",
            seed=3,
            context={"check": "positive-exclusion"},
        )
        self.assertEqual(error.message, "split audit failed")
        self.assertEqual(error.context["dataset"], "cora")
        self.assertEqual(error.context["check"], "positive-exclusion")
        self.assertIn("seed=3", str(error))
        with self.assertRaises(TypeError):
            error.context["dataset"] = "citeseer"


if __name__ == "__main__":
    unittest.main()
