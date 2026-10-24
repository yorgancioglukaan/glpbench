import sys
import unittest
from pathlib import Path
from types import MappingProxyType

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
    TrainingLoss,
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
    ModelAdapter,
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


class _ModelAdapter:
    name = "toy-model"
    parameter_defaults = MappingProxyType({})
    capabilities = ModelCapabilities()

    def validate_parameters(self, parameters):
        return MappingProxyType(dict(parameters))

    def build(self, train_data, parameters):
        return torch.nn.Linear(2, 2)

    def compute_training_loss(
        self, model, train_data, edge_label_index, edge_label, criterion
    ):
        loss = model.weight.sum()
        return TrainingLoss(loss, {"reconstruction": loss})

    def score_edges(self, model, data, edge_index):
        return torch.empty(edge_index.size(1))

    def result_metadata(self, parameters):
        return MappingProxyType(dict(parameters))


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
            (_ModelAdapter(), ModelAdapter),
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

        class IncompleteAdapter:
            name = "incomplete"
            capabilities = ModelCapabilities()

        self.assertNotIsInstance(IncompleteAdapter(), ModelAdapter)

    def test_trainable_and_non_trainable_adapters_satisfy_boundary(self):
        self.assertIsInstance(_ModelAdapter(), ModelAdapter)

        class NonTrainableAdapter(_ModelAdapter):
            capabilities = ModelCapabilities(trainable=False)

        self.assertIsInstance(NonTrainableAdapter(), ModelAdapter)

    def test_adapter_parameter_and_scoring_contract(self):
        adapter = _ModelAdapter()
        parameters = adapter.validate_parameters({"hidden_channels": 8})
        with self.assertRaises(TypeError):
            parameters["hidden_channels"] = 16
        model = adapter.build(None, parameters)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        self.assertEqual(
            adapter.score_edges(model, None, edge_index).shape, (2,)
        )

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
