"""Tests for declarative model--protocol compatibility checks."""

import unittest

from glpbench.contracts import (
    CandidateOrientation,
    DatasetMetadata,
    EndpointMode,
    ModelCapabilities,
    ProtocolRequirements,
)
from glpbench.evaluation import (
    validate_model_protocol_compatibility,
    validate_registered_compatibility,
)
from glpbench.exceptions import CompatibilityError
from glpbench.models import available_model_types
from glpbench.protocols import (
    available_split_protocols,
    get_split_protocol_requirements,
)


def _metadata(*, features=True, directed=False):
    return DatasetMetadata(
        name="toy",
        family="synthetic",
        num_nodes=4,
        num_edges=6,
        directed=directed,
        has_node_features=features,
        num_node_features=2 if features else 0,
    )


class TestCompatibilityValidation(unittest.TestCase):
    def test_builtin_models_support_current_protocols(self):
        for model_name in available_model_types():
            for protocol_name in available_split_protocols():
                with self.subTest(model=model_name, protocol=protocol_name):
                    validate_registered_compatibility(
                        model_name,
                        protocol_name,
                        _metadata(),
                        device="cpu",
                    )

    def test_builtin_protocols_declare_requirements(self):
        for name in available_split_protocols():
            with self.subTest(protocol=name):
                self.assertIsInstance(
                    get_split_protocol_requirements(name),
                    ProtocolRequirements,
                )

    def test_missing_endpoint_support_is_actionable(self):
        capabilities = ModelCapabilities(
            supported_endpoint_modes=(EndpointMode.VISIBLE_VISIBLE,),
        )
        requirements = ProtocolRequirements(
            endpoint_modes=(EndpointMode.VISIBLE_UNSEEN,),
            candidate_orientations=(
                CandidateOrientation.VISIBLE_SOURCE_QUERY_TARGET,
            ),
        )
        with self.assertRaisesRegex(
            CompatibilityError, "visible-unseen"
        ) as raised:
            validate_model_protocol_compatibility(
                capabilities,
                requirements,
                _metadata(),
                device="cpu",
                model_name="transductive-only",
                protocol_name="semi-inductive",
            )
        self.assertEqual(
            raised.exception.context["protocol"], "semi-inductive"
        )

    def test_missing_features_and_unsupported_device_are_reported_together(self):
        requirements = ProtocolRequirements(
            endpoint_modes=(EndpointMode.VISIBLE_VISIBLE,),
            candidate_orientations=(CandidateOrientation.UNDIRECTED,),
        )
        with self.assertRaises(CompatibilityError) as raised:
            validate_model_protocol_compatibility(
                ModelCapabilities(),
                requirements,
                _metadata(features=False),
                device="meta",
                model_name="feature-model",
                protocol_name="transductive",
            )
        message = str(raised.exception)
        self.assertIn("requires node features", message)
        self.assertIn("device 'meta' is unsupported", message)
        self.assertEqual(len(raised.exception.context["failures"]), 2)

    def test_graph_label_auxiliary_and_orientation_failures_are_aggregated(self):
        capabilities = ModelCapabilities(
            requires_labels=True,
            requires_auxiliary_data=True,
            supported_candidate_orientations=(
                CandidateOrientation.UNDIRECTED,
            ),
        )
        requirements = ProtocolRequirements(
            endpoint_modes=(EndpointMode.VISIBLE_VISIBLE,),
            candidate_orientations=(
                CandidateOrientation.VISIBLE_SOURCE_QUERY_TARGET,
            ),
            provides_message_passing_graph=False,
            provides_edge_labels=False,
            provides_auxiliary_data=False,
        )
        with self.assertRaises(CompatibilityError) as raised:
            validate_model_protocol_compatibility(
                capabilities,
                requirements,
                _metadata(directed=True),
                device="cpu",
                model_name="demanding-model",
                protocol_name="restricted",
            )
        message = str(raised.exception)
        for expected in (
            "candidate orientation",
            "message-passing graph",
            "labeled candidate edges",
            "auxiliary protocol data",
            "undirected graph",
        ):
            self.assertIn(expected, message)

    def test_invalid_contract_arguments_are_rejected(self):
        requirements = get_split_protocol_requirements("transductive")
        with self.assertRaisesRegex(TypeError, "ModelCapabilities"):
            validate_model_protocol_compatibility(
                object(), requirements, _metadata(), device="cpu"
            )
        with self.assertRaisesRegex(TypeError, "DatasetMetadata"):
            validate_model_protocol_compatibility(
                ModelCapabilities(), requirements, object(), device="cpu"
            )


if __name__ == "__main__":
    unittest.main()
