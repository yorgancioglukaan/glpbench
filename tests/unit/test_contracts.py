import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch
from torch_geometric.data import Data


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.contracts import (  # noqa: E402
    CandidateOrientation,
    DatasetMetadata,
    EndpointMode,
    EvaluationResult,
    LoadedDataset,
    ModelCapabilities,
    ProtocolMetadata,
    ProtocolRequirements,
    ResolvedExperimentSpec,
    SplitBundle,
    TrainingLoss,
    TrainingPolicy,
    TrainingResult,
)


def _graph():
    return Data(
        x=torch.ones((3, 2)),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        num_nodes=3,
    )


def _dataset_metadata():
    return DatasetMetadata(
        name="toy",
        family="synthetic",
        num_nodes=3,
        num_edges=4,
        directed=False,
        has_node_features=True,
        num_node_features=2,
    )


class TestDatasetContracts(unittest.TestCase):
    def test_dataset_metadata_is_normalized_and_immutable(self):
        metadata = DatasetMetadata(
            name="toy",
            family="synthetic",
            num_nodes=3,
            num_edges=4,
            directed=False,
            has_node_features=True,
            num_node_features=2,
            loader_preprocessing=["coalesce", "normalize"],
        )
        self.assertEqual(metadata.loader_preprocessing, ("coalesce", "normalize"))
        with self.assertRaises(FrozenInstanceError):
            metadata.name = "changed"

    def test_dataset_metadata_rejects_inconsistent_feature_description(self):
        with self.assertRaisesRegex(ValueError, "must be zero"):
            DatasetMetadata(
                name="toy",
                family="synthetic",
                num_nodes=3,
                num_edges=4,
                directed=False,
                has_node_features=False,
                num_node_features=2,
            )

    def test_loaded_dataset_preserves_native_pyg_data(self):
        graph = _graph()
        loaded = LoadedDataset(graph, _dataset_metadata())
        self.assertIs(loaded.data, graph)

    def test_loaded_dataset_checks_shape_and_counts(self):
        graph = _graph()
        bad_counts = DatasetMetadata(
            name="toy",
            family="synthetic",
            num_nodes=4,
            num_edges=4,
            directed=False,
            has_node_features=True,
            num_node_features=2,
        )
        with self.assertRaisesRegex(ValueError, "num_nodes"):
            LoadedDataset(graph, bad_counts)


class TestProtocolContracts(unittest.TestCase):
    def test_protocol_metadata_recursively_freezes_values(self):
        metadata = ProtocolMetadata(
            name="semi-inductive",
            schema_version=1,
            seed=7,
            parameters={"ratios": [0.1, 0.2]},
            requested_counts={"queries": 2},
        )
        self.assertEqual(metadata.parameters["ratios"], (0.1, 0.2))
        with self.assertRaises(TypeError):
            metadata.parameters["seed"] = 9

    def test_split_bundle_normalizes_queries_and_freezes_tests(self):
        graph = _graph()
        bundle = SplitBundle(
            train=graph,
            validation=graph,
            tests={"transductive": graph},
            metadata=ProtocolMetadata(name="paired", schema_version=1),
            validation_query_nodes=(2,),
            test_query_nodes=(1, 0),
            visible_node_mask=torch.tensor([True, True, False]),
        )
        self.assertEqual(bundle.test_query_nodes, (0, 1))
        with self.assertRaises(TypeError):
            bundle.tests["inductive"] = graph

    def test_split_bundle_rejects_overlapping_queries(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            SplitBundle(
                train=_graph(),
                validation=None,
                tests={},
                metadata=ProtocolMetadata(name="paired", schema_version=1),
                validation_query_nodes=(1,),
                test_query_nodes=(1,),
            )

    def test_split_bundle_checks_visibility_mask(self):
        with self.assertRaisesRegex(ValueError, "length"):
            SplitBundle(
                train=_graph(),
                validation=None,
                tests={},
                metadata=ProtocolMetadata(name="paired", schema_version=1),
                visible_node_mask=torch.tensor([True, False]),
            )


class TestModelAndTrainingContracts(unittest.TestCase):
    def test_model_capabilities_normalizes_devices(self):
        capabilities = ModelCapabilities(supported_devices=["cpu", "mps"])
        self.assertEqual(capabilities.supported_devices, ("cpu", "mps"))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            ModelCapabilities(supported_devices=("cpu", "cpu"))

    def test_compatibility_contracts_normalize_enum_values(self):
        capabilities = ModelCapabilities(
            supported_endpoint_modes=["visible-visible"],
            supported_candidate_orientations=["undirected"],
        )
        self.assertEqual(
            capabilities.supported_endpoint_modes,
            (EndpointMode.VISIBLE_VISIBLE,),
        )
        requirements = ProtocolRequirements(
            endpoint_modes=["visible-unseen"],
            candidate_orientations=["visible-source-query-target"],
        )
        self.assertEqual(
            requirements.candidate_orientations,
            (CandidateOrientation.VISIBLE_SOURCE_QUERY_TARGET,),
        )

    def test_compatibility_contracts_reject_duplicate_modes(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            ModelCapabilities(
                supported_endpoint_modes=(
                    EndpointMode.VISIBLE_VISIBLE,
                    EndpointMode.VISIBLE_VISIBLE,
                )
            )

    def test_training_loss_freezes_valid_scalar_components(self):
        total = torch.tensor(1.5, requires_grad=True)
        reconstruction = torch.tensor(1.0)
        loss = TrainingLoss(
            total=total,
            components={"reconstruction": reconstruction},
        )
        self.assertIs(loss.total, total)
        self.assertIs(loss.components["reconstruction"], reconstruction)
        with self.assertRaises(TypeError):
            loss.components["regularization"] = torch.tensor(0.5)

    def test_training_loss_rejects_malformed_values(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            TrainingLoss(total=1.0)
        with self.assertRaisesRegex(ValueError, "scalar"):
            TrainingLoss(total=torch.ones(2))
        with self.assertRaisesRegex(ValueError, "component name"):
            TrainingLoss(torch.tensor(1.0), {"": torch.tensor(1.0)})
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            TrainingLoss(torch.tensor(1.0), {"reconstruction": 1.0})
        with self.assertRaisesRegex(ValueError, "scalar"):
            TrainingLoss(
                torch.tensor(1.0), {"reconstruction": torch.ones(2)}
            )

    def test_training_policy_freezes_nested_parameters(self):
        policy = TrainingPolicy(
            max_epochs=100,
            optimizer_parameters={"betas": [0.9, 0.999]},
            early_stopping_patience=10,
        )
        self.assertEqual(policy.optimizer_parameters["betas"], (0.9, 0.999))
        with self.assertRaises(TypeError):
            policy.optimizer_parameters["lr"] = 0.01

    def test_training_policy_rejects_invalid_epoch_settings(self):
        with self.assertRaisesRegex(ValueError, "between zero and max_epochs"):
            TrainingPolicy(max_epochs=10, early_stopping_minimum_epochs=11)

    def test_training_result_accepts_signed_selection_metrics(self):
        result = TrainingResult(
            epochs_executed=4,
            final_epoch=4,
            best_epoch=2,
            selected_epoch=2,
            checkpoint_strategy="best-validation",
            selection_metric="correlation",
            best_validation_score=-0.2,
            stopped_early=True,
            training_time_seconds=1.5,
            trainable_parameter_count=12,
            loss_history=[1.0, 0.5],
        )
        self.assertEqual(result.loss_history, (1.0, 0.5))
        with self.assertRaises(FrozenInstanceError):
            result.best_epoch = 3

    def test_training_result_rejects_epoch_outside_execution(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TrainingResult(
                epochs_executed=2,
                final_epoch=2,
                best_epoch=3,
                selected_epoch=2,
                checkpoint_strategy="best-validation",
                selection_metric="roc_auc",
                best_validation_score=0.8,
                stopped_early=False,
                training_time_seconds=1.0,
                trainable_parameter_count=12,
            )


class TestExperimentAndEvaluationContracts(unittest.TestCase):
    def test_evaluation_result_freezes_metrics(self):
        result = EvaluationResult(
            setting="semi-inductive",
            metrics={"roc_auc": 0.8},
            positive_count=5,
            negative_count=5,
        )
        with self.assertRaises(TypeError):
            result.metrics["average_precision"] = 0.7

    def test_evaluation_result_rejects_nonfinite_metric(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            EvaluationResult(
                setting="transductive",
                metrics={"roc_auc": float("nan")},
                positive_count=1,
                negative_count=1,
            )

    def test_resolved_spec_freezes_all_parameter_groups(self):
        spec = ResolvedExperimentSpec(
            schema_version=1,
            dataset="cora",
            model_type="gae",
            protocol="paired",
            seed=42,
            device="cpu",
            training_policy=TrainingPolicy(max_epochs=10),
            model_parameters={"hidden_channels": [16, 32]},
        )
        self.assertEqual(spec.model_parameters["hidden_channels"], (16, 32))
        with self.assertRaises(TypeError):
            spec.model_parameters["dropout"] = 0.5


if __name__ == "__main__":
    unittest.main()
    EndpointMode,
