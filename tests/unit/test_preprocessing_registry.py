"""Tests for registered and composed feature preprocessing."""

import unittest
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from glpbench.interfaces import FeaturePreprocessor
from glpbench.preprocessing import (
    PreprocessingPipeline,
    PreprocessingStep,
    available_preprocessors,
    create_preprocessor,
    fit_reducer,
    normalize_graph_features,
    register_preprocessor,
    transform_features,
)
import glpbench.preprocessing.registry as registry


class _AffinePreprocessor:
    requires_fitting = False
    supports_sparse_inputs = False
    uses_randomness = False
    output = "features"

    def __init__(self, name, scale=1.0, offset=0.0, calls=None):
        self.name = name
        self.scale = scale
        self.offset = offset
        self.calls = calls

    def fit(self, x, fit_mask, seed=None):
        if self.calls is not None:
            self.calls.append((self.name, x.clone(), fit_mask.clone()))
        return self

    def transform(self, x):
        return x * self.scale + self.offset


class TestPreprocessorRegistry(unittest.TestCase):
    def setUp(self):
        self._factories = dict(registry._PREPROCESSOR_FACTORIES)
        self._lookup = dict(registry._NAME_LOOKUP)

    def tearDown(self):
        registry._PREPROCESSOR_FACTORIES.clear()
        registry._PREPROCESSOR_FACTORIES.update(self._factories)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)

    def test_builtin_names_and_conformance(self):
        self.assertEqual(available_preprocessors(), (
            "identity",
            "normalize-features",
            "pca",
            "sparse-random-projection",
            "truncated-svd",
        ))
        for name in available_preprocessors():
            parameters = {} if name in {"identity", "normalize-features"} else {
                "n_components": 2
            }
            with self.subTest(name=name):
                self.assertIsInstance(
                    create_preprocessor(name, **parameters), FeaturePreprocessor
                )

    def test_alias_duplicate_replacement_and_fresh_instances(self):
        register_preprocessor(
            "affine",
            lambda **kwargs: _AffinePreprocessor("affine", **kwargs),
            aliases=("Affine Transform",),
        )
        first = create_preprocessor("affine-transform", scale=2)
        second = create_preprocessor("AFFINE", scale=3)
        self.assertIsNot(first, second)
        self.assertEqual((first.scale, second.scale), (2, 3))
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_preprocessor("affine", lambda: first)
        register_preprocessor(
            "affine", lambda: _AffinePreprocessor("affine"), replace=True
        )

    def test_invalid_factory_output_and_unknown_name(self):
        register_preprocessor("invalid", lambda: object())
        with self.assertRaisesRegex(TypeError, "FeaturePreprocessor"):
            create_preprocessor("invalid")
        with self.assertRaisesRegex(ValueError, "Available preprocessors"):
            create_preprocessor("missing")


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self._factories = dict(registry._PREPROCESSOR_FACTORIES)
        self._lookup = dict(registry._NAME_LOOKUP)

    def tearDown(self):
        registry._PREPROCESSOR_FACTORIES.clear()
        registry._PREPROCESSOR_FACTORIES.update(self._factories)
        registry._NAME_LOOKUP.clear()
        registry._NAME_LOOKUP.update(self._lookup)

    @staticmethod
    def _features():
        return torch.tensor([
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 2.0],
            [3.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ])

    def test_identity_is_exact_and_does_not_mutate_input(self):
        features = self._features()
        original = features.clone()
        result = PreprocessingPipeline([
            PreprocessingStep("identity")
        ]).fit_transform(features, torch.tensor([True, True, False, False]))

        self.assertTrue(torch.equal(result.features, original))
        self.assertIsNone(result.reduced_features)
        self.assertIsNone(result.fit_time_seconds)
        result.features[0, 0] = 99
        self.assertTrue(torch.equal(features, original))

    def test_order_is_respected_and_fit_mask_is_pipeline_controlled(self):
        calls = []
        register_preprocessor(
            "double",
            lambda: _AffinePreprocessor("double", scale=2, calls=calls),
        )
        register_preprocessor(
            "plus-one",
            lambda: _AffinePreprocessor("plus-one", offset=1, calls=calls),
        )
        features = self._features()
        mask = torch.tensor([True, False, True, False])
        result = PreprocessingPipeline([
            PreprocessingStep("double"),
            PreprocessingStep("plus-one"),
        ]).fit_transform(features, mask)

        self.assertTrue(torch.equal(result.features, features * 2 + 1))
        self.assertEqual([call[0] for call in calls], ["double", "plus-one"])
        self.assertTrue(torch.equal(calls[0][2], mask))
        self.assertTrue(torch.equal(calls[1][2], mask))
        self.assertTrue(torch.equal(calls[1][1], features * 2))

    def test_normalization_and_pca_match_existing_primitives(self):
        features = self._features()
        mask = torch.tensor([True, True, True, False])
        normalized = normalize_graph_features(
            Data(x=features, num_nodes=features.size(0))
        ).x
        reducer = fit_reducer(
            normalized,
            fit_nodes=mask,
            method="pca",
            n_components=2,
            random_state=11,
        )
        expected_reduced = transform_features(normalized, reducer)

        result = PreprocessingPipeline([
            PreprocessingStep("normalize-features"),
            PreprocessingStep("pca", {"n_components": 2}),
        ]).fit_transform(features, mask, seed=11)

        self.assertTrue(torch.equal(result.features, normalized))
        self.assertTrue(torch.allclose(
            result.reduced_features, expected_reduced, atol=1e-6
        ))
        self.assertIsNotNone(result.fit_time_seconds)

    def test_reduction_must_be_final(self):
        pipeline = PreprocessingPipeline([
            PreprocessingStep("pca", {"n_components": 2}),
            PreprocessingStep("normalize-features"),
        ])
        with self.assertRaisesRegex(ValueError, "final"):
            pipeline.fit_transform(
                self._features(), torch.tensor([True, True, True, False])
            )


if __name__ == "__main__":
    unittest.main()
