import unittest
from unittest.mock import patch
import argparse
import copy
import csv
from dataclasses import replace
import json
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import yaml
import utils
import datasets
import main
import benchmark
import search
import feature_reduction
import experiment_pipeline
from experiment_logging import (
    CSVResultLogger,
    ExperimentResult,
    make_configuration_id,
)
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, SAGEConv
import link_prediction


class TestFeatureReduction(unittest.TestCase):
    class SpyReducer:
        def __init__(self, n_components, random_state=None, **kwargs):
            self.n_components = n_components
            self.random_state = random_state
            self.kwargs = kwargs
            self.fit_calls = []
            self.transform_calls = []

        def fit(self, features):
            self.fit_calls.append(features.copy())
            return self

        def transform(self, features):
            self.transform_calls.append(features.copy())
            return features[:, :self.n_components]

    def test_fit_uses_selected_nodes_and_transform_uses_all_nodes(self):
        features = torch.tensor([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ])
        fit_mask = torch.tensor([True, False, True, False])
        with patch.dict(
            feature_reduction._REDUCER_FACTORIES,
            {"spy": self.SpyReducer},
            clear=True,
        ):
            reducer = feature_reduction.fit_reducer(
                features,
                fit_nodes=fit_mask,
                method="spy",
                n_components=2,
                random_state=7,
                custom_option=True,
            )
            transformed = feature_reduction.transform_features(features, reducer)

        self.assertEqual(len(reducer.fit_calls), 1)
        self.assertTrue(
            torch.equal(torch.from_numpy(reducer.fit_calls[0]), features[fit_mask])
        )
        self.assertEqual(len(reducer.transform_calls), 1)
        self.assertTrue(
            torch.equal(torch.from_numpy(reducer.transform_calls[0]), features)
        )
        self.assertEqual(transformed.shape, (4, 2))
        self.assertEqual(reducer.random_state, 7)
        self.assertEqual(reducer.kwargs, {"custom_option": True})

    def test_update_uses_transform_without_refitting(self):
        data = Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        reducer = self.SpyReducer(n_components=1)
        reducer.fit(data.x.numpy())
        data.feature_reducer = reducer

        utils.update_reduced_features(data)

        self.assertEqual(len(reducer.fit_calls), 1)
        self.assertEqual(len(reducer.transform_calls), 1)
        self.assertTrue(torch.equal(data.x_reduced, data.x[:, :1]))

    def test_registry_and_invalid_method(self):
        self.assertEqual(feature_reduction.available_reduction_methods(), (
            "pca", "sparse-random-projection", "truncated-svd"
        ))
        with self.assertRaisesRegex(ValueError, "Unknown reduction method"):
            feature_reduction.fit_reducer(torch.ones(3, 2), method="unknown")

    def test_registered_reducers_transform_unseen_nodes(self):
        features = torch.tensor([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [1.0, 2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 1.0],
        ])
        fit_mask = torch.tensor([True, True, True, True, False, False])
        for method in ("truncated-svd", "sparse-random-projection"):
            with self.subTest(method=method):
                reducer = feature_reduction.fit_reducer(
                    features,
                    fit_nodes=fit_mask,
                    method=method,
                    n_components=2,
                    random_state=11,
                )
                transformed = feature_reduction.transform_features(
                    features, reducer
                )
                self.assertEqual(transformed.shape, (6, 2))
                self.assertTrue(torch.isfinite(transformed).all())
                self.assertEqual(reducer.reduction_method, method)

    def test_stochastic_reducers_are_reproducible(self):
        features = torch.arange(40, dtype=torch.float).reshape(8, 5)
        for method in ("truncated-svd", "sparse-random-projection"):
            with self.subTest(method=method):
                first = feature_reduction.fit_reducer(
                    features, method=method, n_components=2, random_state=5
                )
                second = feature_reduction.fit_reducer(
                    features, method=method, n_components=2, random_state=5
                )
                self.assertTrue(torch.allclose(
                    feature_reduction.transform_features(features, first),
                    feature_reduction.transform_features(features, second),
                ))

    def test_boolean_fit_mask_must_match_node_count(self):
        with self.assertRaisesRegex(ValueError, "one entry per node"):
            feature_reduction.fit_reducer(
                torch.ones(3, 2), fit_nodes=torch.tensor([True, False])
            )

class TestInductiveSplit(unittest.TestCase):
    @staticmethod
    def _undirected_data(num_nodes, edges):
        directed_edges = edges + [(target, source) for source, target in edges]
        return Data(
            edge_index=torch.tensor(directed_edges, dtype=torch.long).t().contiguous(),
            num_nodes=num_nodes,
        )

    def test_automatic_query_selection_is_reproducible_and_sorted(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 0), (0, 4), (2, 6)],
        )
        first = utils.select_query_nodes(data, query_size=3, seed=17)
        second = utils.select_query_nodes(data, query_size=3, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 3)
        self.assertNotEqual(
            first,
            utils.select_query_nodes(data, query_size=3, seed=18),
        )

    def test_automatic_query_selection_uses_fraction_and_excludes_isolates(self):
        data = self._undirected_data(
            6, [(0, 1), (1, 2), (2, 3), (3, 4)]
        )
        query = utils.select_query_nodes(
            data, query_fraction=0.4, seed=5
        )

        self.assertEqual(len(query), 2)
        self.assertNotIn(5, query)

    def test_automatic_query_selection_preserves_global_rng_state(self):
        data = self._undirected_data(
            6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        )
        random.seed(1234)
        state = random.getstate()

        utils.select_query_nodes(data, query_size=2, seed=9)

        self.assertEqual(random.getstate(), state)

    def test_every_automatic_query_has_a_visible_neighbor(self):
        data = self._undirected_data(
            7, [(0, 1), (2, 3), (3, 4), (4, 5), (5, 6), (6, 2)]
        )
        query = utils.select_query_nodes(data, query_size=2, seed=2)
        query_set = set(query)
        neighbors = {node: set() for node in range(data.num_nodes)}
        for source, target in data.edge_index.t().tolist():
            neighbors[source].add(target)

        self.assertTrue(all(neighbors[node] - query_set for node in query))

    def test_joint_query_selection_is_disjoint_reproducible_and_visible(self):
        data = self._undirected_data(
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 8), (8, 9), (9, 0), (0, 5), (2, 7)],
        )
        random.seed(1234)
        state = random.getstate()

        first = utils.select_query_node_sets(
            data, query_size=2, validation_query_size=2, seed=17
        )
        second = utils.select_query_node_sets(
            data, query_size=2, validation_query_size=2, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)
        self.assertTrue(set(first.validation).isdisjoint(first.test))
        self.assertEqual(list(first.validation), sorted(first.validation))
        self.assertEqual(list(first.test), sorted(first.test))
        query_union = set(first.validation) | set(first.test)
        neighbors = utils._node_neighbors(data)
        self.assertTrue(all(
            neighbors[node] - query_union for node in query_union
        ))

    def test_incremental_query_selection_succeeds_on_sparse_graph(self):
        edges = [(2 * index, 2 * index + 1) for index in range(1000)]
        data = self._undirected_data(2000, edges)

        selected = utils.select_query_node_sets(
            data,
            query_fraction=0.05,
            validation_query_fraction=0.05,
            seed=23,
            max_attempts=1,
        )

        query_union = set(selected.validation) | set(selected.test)
        neighbors = utils._node_neighbors(data)
        self.assertEqual(len(selected.test), 100)
        self.assertEqual(len(selected.validation), 100)
        self.assertTrue(set(selected.validation).isdisjoint(selected.test))
        self.assertTrue(all(
            neighbors[node] - query_union for node in query_union
        ))

    def test_joint_query_selection_preserves_explicit_queries(self):
        data = self._undirected_data(
            6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        )

        selected = utils.select_query_node_sets(
            data, test_query=[0, 2], validation_query=[4], seed=7
        )

        self.assertEqual(selected.test, (0, 2))
        self.assertEqual(selected.validation, (4,))

    def test_joint_query_selection_rejects_overlap_and_infeasible_counts(self):
        data = self._undirected_data(
            5, [(0, 1), (1, 2), (2, 3), (3, 4)]
        )
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            utils.select_query_node_sets(
                data, test_query=[1], validation_query=[1], seed=2
            )
        with self.assertRaisesRegex(ValueError, "no feasible"):
            utils.select_query_node_sets(
                data, query_size=3, validation_query_size=2, seed=2
            )

    def test_query_selection_validates_configuration_and_explicit_queries(self):
        data = self._undirected_data(
            5, [(0, 1), (1, 2), (2, 3), (3, 4)]
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            utils.select_query_nodes(data, seed=1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            utils.select_query_nodes(
                data, query_size=1, query_fraction=0.2, seed=1
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            utils.inductive_split(data, query=[1, 1])
        with self.assertRaisesRegex(ValueError, "outside the graph"):
            utils.inductive_split(data, query=[5])
        with self.assertRaisesRegex(ValueError, "training-visible neighbor"):
            utils.inductive_split(
                data, query=[0, 1], validation_query=[3, 4]
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            utils.select_query_nodes(data, query_size=0, seed=1)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            utils.select_query_nodes(data, query_fraction=1.0, seed=1)

        two_node_graph = self._undirected_data(2, [(0, 1)])
        with self.assertRaisesRegex(ValueError, "retains a visible neighbor"):
            utils.select_query_nodes(
                two_node_graph, query_size=2, seed=1, max_attempts=2
            )

    def test_split_functions_support_automatic_query_selection(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7)],
        )
        train, val, test_transductive, test_inductive = (
            utils.generate_comparable_splits(
                data, query_size=1, seed=11, train_ratio=0.75
            )
        )

        expected_queries = utils.select_query_node_sets(
            data, query_size=1, seed=11
        )
        for split in (train, val, test_transductive, test_inductive):
            actual_test = (
                split.test_query_mask.nonzero(as_tuple=False).view(-1).tolist()
            )
            actual_validation = (
                split.validation_query_mask.nonzero(as_tuple=False).view(-1).tolist()
            )
            self.assertEqual(actual_test, list(expected_queries.test))
            self.assertEqual(
                actual_validation, list(expected_queries.validation)
            )
        self.assertTrue(torch.equal(
            test_transductive.edge_index, test_inductive.edge_index
        ))
        self.assertEqual(
            test_transductive.edge_label_index.size(1),
            test_inductive.edge_label_index.size(1),
        )

    def test_comparable_splits_support_named_and_legacy_access(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7)],
        )

        splits = utils.generate_comparable_splits(
            data, query=[7], validation_query=[6], seed=3, train_ratio=0.75
        )
        unpacked = tuple(splits)

        self.assertIsInstance(splits, utils.ComparableSplits)
        self.assertIs(splits.train, unpacked[0])
        self.assertIs(splits.validation, unpacked[1])
        self.assertIs(splits.test_transductive, unpacked[2])
        self.assertIs(splits.test_inductive, unpacked[3])

    def test_split_metadata_exposes_explicit_node_and_split_roles(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7)],
        )
        splits = utils.generate_comparable_splits(
            data, query=[7], validation_query=[6], seed=3, train_ratio=0.75
        )
        expected_validation_queries = torch.tensor(
            [False, False, False, False, False, False, True, False]
        )
        expected_test_queries = torch.tensor(
            [False, False, False, False, False, False, False, True]
        )

        expected_roles = (
            (splits.train, "train", "transductive"),
            (splits.validation, "validation", "semi-inductive"),
            (splits.test_transductive, "test", "transductive"),
            (splits.test_inductive, "test", "semi-inductive"),
        )
        for split, stage, setting in expected_roles:
            self.assertTrue(torch.equal(
                split.validation_query_mask, expected_validation_queries
            ))
            self.assertTrue(torch.equal(
                split.test_query_mask, expected_test_queries
            ))
            self.assertTrue(torch.equal(
                split.visible_node_mask,
                ~(expected_validation_queries | expected_test_queries),
            ))
            self.assertTrue(torch.equal(
                split.query_mask,
                split.validation_query_mask | split.test_query_mask,
            ))
            self.assertTrue(torch.equal(
                split.seen_node_mask, split.visible_node_mask
            ))
            self.assertEqual(split.split_stage, stage)
            self.assertEqual(split.split_setting, setting)

    def test_node_role_metadata_rejects_overlapping_query_sets(self):
        split = Data(num_nodes=3)
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            utils._attach_split_metadata(
                (("validation", "validation", "semi-inductive", split),),
                utils.QueryNodeSets(validation=(1,), test=(1,)),
                num_nodes=3,
                device=torch.device("cpu"),
            )

    def test_comparable_splits_are_reproducible_for_a_fixed_seed(self):
        data = self._undirected_data(
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 5), (0, 6), (2, 7), (1, 8), (3, 9)],
        )
        kwargs = dict(
            query=[8, 9], validation_query=[6, 7],
            train_ratio=0.75, seed=21,
        )

        first = utils.generate_comparable_splits(data, **kwargs)
        second = utils.generate_comparable_splits(data, **kwargs)

        for first_split, second_split in zip(first, second):
            for attribute in (
                "edge_index", "edge_label_index", "edge_label",
                "visible_node_mask", "validation_query_mask",
                "test_query_mask",
            ):
                self.assertTrue(torch.equal(
                    getattr(first_split, attribute),
                    getattr(second_split, attribute),
                ))

    def test_comparable_split_generation_runs_end_to_end_audit(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 6), (3, 7)],
        )
        with patch(
            "utils.audit_comparable_splits",
            wraps=utils.audit_comparable_splits,
        ) as audit:
            splits = utils.generate_comparable_splits(
                data, query=[7], validation_query=[6], seed=4
            )

        audit.assert_called_once()
        self.assertIs(audit.call_args.args[0], data)
        self.assertIs(audit.call_args.args[1], splits)

    def test_audit_rejects_corrupted_split_invariants(self):
        data = self._undirected_data(
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 5), (0, 6), (2, 7), (1, 8), (3, 9)],
        )
        clean = utils.generate_comparable_splits(
            data, query=[8, 9], validation_query=[6, 7], seed=9,
            train_ratio=0.75,
        )

        validation = clean.validation.clone()
        validation.edge_index = torch.cat([
            validation.edge_index,
            torch.tensor([[0, 6], [6, 0]]),
        ], dim=1)
        with self.assertRaisesRegex(ValueError, "query node"):
            utils.audit_comparable_splits(
                data, clean._replace(validation=validation)
            )

        inductive = clean.test_inductive.clone()
        negative_index = int((inductive.edge_label == 1).sum())
        inductive.edge_label_index[:, negative_index] = data.edge_index[:, 0]
        with self.assertRaisesRegex(ValueError, "negative edge that is positive"):
            utils.audit_comparable_splits(
                data, clean._replace(test_inductive=inductive)
            )

        transductive = clean.test_transductive.clone()
        transductive.edge_label_index = transductive.edge_label_index[:, :-1]
        transductive.edge_label = transductive.edge_label[:-1]
        with self.assertRaisesRegex(ValueError, "different labeled-edge counts"):
            utils.audit_comparable_splits(
                data, clean._replace(test_transductive=transductive)
            )

        validation = clean.validation.clone()
        validation_positive = validation.edge_label == 1
        validation.edge_label_index[:, validation_positive] = (
            clean.test_inductive.edge_label_index[
                :, clean.test_inductive.edge_label == 1
            ][:, :int(validation_positive.sum())]
        )
        with self.assertRaisesRegex(ValueError, "Positive edges overlap"):
            utils.audit_comparable_splits(
                data, clean._replace(validation=validation)
            )

    def test_query_to_query_edges_are_excluded_from_all_split_roles(self):
        data = self._undirected_data(
            6,
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (4, 5)],
        )
        splits = utils.generate_comparable_splits(
            data, query=[5], validation_query=[4], seed=2,
            train_ratio=0.75,
        )
        query_pair = (4, 5)

        for split in splits:
            represented = (
                utils._canonical_edge_pairs(split.edge_index)
                | utils._canonical_edge_pairs(split.edge_label_index)
            )
            self.assertNotIn(query_pair, represented)
        positive_sets = [
            utils._canonical_edge_pairs(
                split.edge_label_index[:, split.edge_label == 1]
            )
            for split in splits
        ]
        for index, left in enumerate(positive_sets):
            for right in positive_sets[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertIsNone(utils.audit_comparable_splits(data, splits))

    def test_split_protocol_rejects_self_loops_and_duplicate_edges(self):
        self_loop = self._undirected_data(
            7,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (2, 2)],
        )
        duplicate = self._undirected_data(
            7,
            [(0, 1), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6)],
        )

        with self.assertRaisesRegex(ValueError, "Self-loops"):
            utils.generate_comparable_splits(
                self_loop, query=[6], validation_query=[5], seed=1
            )
        with self.assertRaisesRegex(ValueError, "Duplicate directed edges"):
            utils.generate_comparable_splits(
                duplicate, query=[6], validation_query=[5], seed=1
            )

    def test_query_int(self):
        data = self._undirected_data(4, [(0, 1), (1, 2), (2, 3)])
        #negative input should throw ValueError
        with self.assertRaises(ValueError):
            utils.inductive_split(data, query = -3)
        #float is invalid type
        with self.assertRaises(TypeError):
            utils.inductive_split(data, query = 1.3)

    def test_query_tensor(self):
        data = self._undirected_data(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        #negative input should throw ValueError
        with self.assertRaises(ValueError):
            utils.inductive_split(data, torch.tensor([1,2,3,-1], dtype=torch.int32))
        #tensor is not 1-d should raise TypeError
        with self.assertRaises(TypeError):
            utils.inductive_split(data, torch.tensor([[1,2],[3,4]],dtype=torch.int64))
        #tensor is not of int type
        with self.assertRaises(TypeError):
            utils.inductive_split(data, torch.tensor([1,2,3,4],dtype=torch.float32))

    def test_query_list(self):
        data = self._undirected_data(
            6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        )
        #
        with self.assertRaises(TypeError):
            utils.inductive_split(data, [[2,3],[3,4]])
        splits = utils.inductive_split(data, [5], validation_query=[4])
        self.assertIsInstance(splits, utils.InductiveSplits)
        train, val, test = splits
        self.assertIs(splits.train, train)
        self.assertIs(splits.validation, val)
        self.assertIs(splits.test_inductive, test)
        expected_query_mask = torch.tensor([
            False, False, False, False, True, True
        ])
        for split, split_type in (
            (train, "train"), (val, "validation"), (test, "inductive")
        ):
            self.assertEqual(split.split_type, split_type)
            self.assertTrue(torch.equal(split.query_mask, expected_query_mask))
            self.assertTrue(torch.equal(split.seen_node_mask, ~expected_query_mask))

    def test_inductive_negatives_do_not_overlap_positive_edges(self):
        data = self._undirected_data(
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 5), (0, 6), (2, 7), (1, 8), (3, 9)],
        )
        validation_query_nodes = {6, 7}
        test_query_nodes = {8, 9}
        query_nodes = validation_query_nodes | test_query_nodes

        train, val, test_transductive, test_inductive = utils.generate_comparable_splits(
            data,
            query=list(test_query_nodes),
            validation_query=list(validation_query_nodes),
            train_ratio=0.75,
            seed=9,
        )

        num_positive = int(test_inductive.edge_label.sum().item())
        negative_edges = test_inductive.edge_label_index[:, num_positive:]
        positive_edges = {
            tuple(sorted(edge)) for edge in data.edge_index.t().tolist()
        }
        validation_negatives = val.edge_label_index[:, val.edge_label == 0]
        transductive_negatives = test_transductive.edge_label_index[
            :, test_transductive.edge_label == 0
        ]

        self.assertTrue(torch.equal(test_inductive.edge_index, test_transductive.edge_index))
        self.assertEqual(
            test_inductive.edge_label_index.size(1),
            test_transductive.edge_label_index.size(1),
        )
        self.assertEqual(negative_edges.size(1), num_positive)
        self.assertEqual(test_transductive.edge_label_index.size(1), 2 * num_positive)
        self.assertTrue(all(
            node not in query_nodes
            for node in test_transductive.edge_label_index.flatten().tolist()
        ))
        self.assertEqual(
            validation_negatives.size(1), int(val.edge_label.sum().item())
        )
        self.assertEqual(
            len({tuple(edge) for edge in validation_negatives.t().tolist()}),
            validation_negatives.size(1),
        )
        for edge in validation_negatives.t().tolist():
            self.assertNotIn(tuple(sorted(edge)), positive_edges)
        for edge in transductive_negatives.t().tolist():
            self.assertNotIn(tuple(sorted(edge)), positive_edges)
        expected_query_mask = torch.tensor(
            [False, False, False, False, False, False, True, True, True, True]
        )
        for split, split_type in (
            (train, "train"),
            (val, "validation"),
            (test_transductive, "transductive"),
            (test_inductive, "inductive"),
        ):
            self.assertEqual(split.split_type, split_type)
            self.assertTrue(torch.equal(split.query_mask, expected_query_mask))
            self.assertTrue(torch.equal(split.seen_node_mask, ~expected_query_mask))
            self.assertTrue(all(
                node not in query_nodes
                for node in split.edge_index.flatten().tolist()
            ))
        for source, target in negative_edges.t().tolist():
            self.assertIn(target, test_query_nodes)
            self.assertNotIn(source, query_nodes)
            self.assertNotEqual(source, target)
            self.assertFalse(source in query_nodes and target in query_nodes)
            self.assertNotIn(tuple(sorted((source, target))), positive_edges)

        validation_positives = val.edge_label_index[:, val.edge_label == 1]
        for source, target in validation_positives.t().tolist():
            self.assertNotIn(source, query_nodes)
            self.assertIn(target, validation_query_nodes)
        for source, target in validation_negatives.t().tolist():
            self.assertNotIn(source, query_nodes)
            self.assertIn(target, validation_query_nodes)

    def test_inductive_negative_sampling_raises_when_no_candidate_exists(self):
        data = self._undirected_data(
            4, [(0, 3), (1, 3), (2, 3), (0, 1), (1, 2), (0, 2)]
        )

        with self.assertRaisesRegex(ValueError, "no eligible non-neighbor"):
            utils._sample_inductive_negative_edges(
                data, torch.tensor([[0], [3]]), query=[3], seed=1
            )


class TestSeenNegativeSampling(unittest.TestCase):
    @staticmethod
    def _undirected_data(num_nodes, edges):
        directed_edges = edges + [(target, source) for source, target in edges]
        return Data(
            edge_index=torch.tensor(
                directed_edges, dtype=torch.long
            ).t().contiguous(),
            num_nodes=num_nodes,
        )

    def test_samples_unique_visible_non_edges_from_original_graph(self):
        data = self._undirected_data(
            7,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (2, 6)],
        )
        seen_mask = torch.tensor(
            [True, True, True, True, True, True, False]
        )

        negatives = utils._sample_seen_negative_edges(
            data, seen_mask, num_samples=5, seed=17
        )

        original_positives = {
            tuple(sorted(edge)) for edge in data.edge_index.t().tolist()
        }
        sampled_pairs = {tuple(edge) for edge in negatives.t().tolist()}
        self.assertEqual(negatives.shape, (2, 5))
        self.assertEqual(len(sampled_pairs), 5)
        self.assertTrue(bool(seen_mask[negatives].all()))
        self.assertTrue(bool((negatives[0] < negatives[1]).all()))
        self.assertTrue(sampled_pairs.isdisjoint(original_positives))

    def test_seed_is_reproducible_without_changing_global_rng_state(self):
        data = self._undirected_data(
            6, [(0, 1), (1, 2), (2, 3), (3, 4)]
        )
        seen_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        torch.manual_seed(1234)
        random.seed(1234)
        initial_torch_state = torch.random.get_rng_state().clone()
        initial_python_state = random.getstate()

        first = utils._sample_seen_negative_edges(
            data, seen_mask, num_samples=4, seed=9
        )
        second = utils._sample_seen_negative_edges(
            data, seen_mask, num_samples=4, seed=9
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(
            torch.random.get_rng_state(), initial_torch_state
        ))
        self.assertEqual(random.getstate(), initial_python_state)

    def test_rejects_requests_larger_than_available_non_edges(self):
        data = self._undirected_data(
            5,
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
             (3, 4)],
        )
        seen_mask = torch.tensor([True, True, True, True, False])

        with self.assertRaisesRegex(ValueError, "only 0 eligible"):
            utils._sample_seen_negative_edges(
                data, seen_mask, num_samples=1, seed=3
            )

    def test_validates_mask_and_supports_zero_samples(self):
        data = self._undirected_data(4, [(0, 1), (1, 2)])
        seen_mask = torch.tensor([True, True, True, False])

        empty = utils._sample_seen_negative_edges(
            data, seen_mask, num_samples=0, seed=1
        )
        self.assertEqual(empty.shape, (2, 0))
        with self.assertRaisesRegex(TypeError, "boolean dtype"):
            utils._sample_seen_negative_edges(
                data, seen_mask.long(), num_samples=1
            )
        with self.assertRaisesRegex(ValueError, "one entry per node"):
            utils._sample_seen_negative_edges(
                data, seen_mask[:-1], num_samples=1
            )

class TestLinkPrediction(unittest.TestCase):
    @staticmethod
    def _training_data():
        return Data(
            x=torch.tensor([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [1.0, 1.0, 0.0],
                [0.5, 0.0, 1.0],
                [0.0, 0.5, 1.0],
            ]),
            x_reduced=torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 1.0],
                [1.0, 0.5],
            ]),
            edge_index=torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4],
                [1, 0, 2, 1, 3, 2, 4, 3],
            ]),
            edge_label_index=torch.tensor([[0, 1], [1, 2]]),
            edge_label=torch.ones(2),
        )

    @classmethod
    def _deal_training_data(cls):
        data = cls._training_data()
        data.edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ])
        data.query_mask = torch.tensor([False, False, False, False, True])
        data.seen_node_mask = ~data.query_mask
        data.split_type = "train"
        return data

    def test_model_construction_for_supported_convolutions(self):
        layer_types = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
            "GATv2Conv": GATv2Conv,
        }
        for model_class in (link_prediction.GAE_Model, link_prediction.AGAE_Model):
            for layer_name, layer_class in layer_types.items():
                with self.subTest(model=model_class.__name__, layer=layer_name):
                    model = model_class(3, 4, 2, conv_layer_type=layer_name)
                    self.assertIsInstance(model.conv1, layer_class)
                    self.assertIsInstance(model.conv2, layer_class)
                    self.assertIsInstance(model, link_prediction.LinkPredictionModel)

    def test_models_share_data_aware_decoder_interface(self):
        data = self._training_data()
        symmetric = link_prediction.GAE_Model(3, 4, 2, conv_layer_type="GCNConv")
        asymmetric = link_prediction.AGAE_Model(3, 4, 2, conv_layer_type="GCNConv")
        z = torch.ones((data.num_nodes, 2))

        symmetric_scores = symmetric.decode_edges(z, data.edge_label_index, data)
        asymmetric_scores = asymmetric.decode_edges(z, data.edge_label_index, data)

        self.assertEqual(symmetric_scores.shape, data.edge_label.shape)
        self.assertEqual(asymmetric_scores.shape, data.edge_label.shape)
        self.assertEqual(symmetric.regularization_loss().item(), 0.0)
        self.assertEqual(asymmetric.regularization_loss().item(), 0.0)

    def test_invalid_model_and_convolution_types(self):
        data = self._training_data()
        with self.assertRaises(ValueError):
            link_prediction.GAE_Model(3, 4, 2, conv_layer_type="InvalidConv")
        with self.assertRaises(ValueError):
            link_prediction.AGAE_Model(3, 4, 2, conv_layer_type="InvalidConv")
        with self.assertRaises(ValueError):
            link_prediction.LinkPrediction(data, model_type="invalid")

    def test_available_model_types_come_from_registry(self):
        self.assertEqual(
            link_prediction.available_model_types(),
            (
                "symmetric", "asymmetric", "mlp", "asymmetric-mlp", "variational",
                "deal", "feature-cosine",
            ),
        )

    def test_model_specs_describe_asymmetric_mlp_capabilities(self):
        spec = link_prediction.get_model_spec("asymmetric-mlp")

        self.assertIs(spec.constructor, link_prediction.MLP_AGAE_Model)
        self.assertTrue(spec.uses_conv_layer)
        self.assertTrue(spec.uses_reduced_features)
        self.assertTrue(spec.supports_agae_regularization)
        self.assertTrue(spec.uses_mlp_decoder)
        with self.assertRaisesRegex(ValueError, "Unexpected model type"):
            link_prediction.get_model_spec("unknown")

    def test_deal_has_structural_embeddings_only_for_seen_nodes(self):
        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="deal", hidden_channels=4, out_channels=2
        )

        structure, attributes = predictor.model.encode(data.x, data.edge_index)

        self.assertEqual(predictor.model.structure_embeddings.num_embeddings, 4)
        self.assertEqual(predictor.model.global_to_seen[4].item(), -1)
        self.assertTrue(torch.equal(structure[4], torch.zeros(2)))
        self.assertEqual(attributes.shape, (data.num_nodes, 2))

    def test_deal_training_uses_three_loss_components(self):
        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="deal", hidden_channels=4, out_channels=2
        )
        negatives = torch.tensor([[0, 3], [2, 0]])
        before = [
            parameter.detach().clone() for parameter in predictor.model.parameters()
        ]

        loss = predictor.train(negative_edge_index=negatives)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(predictor.last_loss_components),
            {"structure", "attribute", "alignment"},
        )
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, predictor.model.parameters())
        ))

    def test_deal_uses_split_specific_score_components(self):
        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="deal",
            hidden_channels=4,
            out_channels=2,
            lambda_ss=2.0,
            lambda_aa=3.0,
            lambda_sa=5.0,
        )
        predictor.model.eval()
        representations = predictor.model.encode(data.x, data.edge_index)

        transductive = data.clone()
        transductive.split_type = "transductive"
        transductive.edge_label_index = torch.tensor([[0], [3]])
        trans_components = predictor.model._component_logits(
            representations, transductive.edge_label_index
        )
        expected_trans = sum(
            weight * score
            for weight, score in zip((2.0, 3.0, 5.0), trans_components)
        ) / 10.0

        inductive = data.clone()
        inductive.split_type = "inductive"
        inductive.edge_label_index = torch.tensor([[0], [4]])
        ind_components = predictor.model._component_logits(
            representations, inductive.edge_label_index
        )
        expected_ind = (
            3.0 * ind_components[1] + 5.0 * ind_components[2]
        ) / 8.0

        self.assertTrue(torch.allclose(
            predictor.test(transductive), expected_trans.sigmoid()
        ))
        self.assertTrue(torch.allclose(
            predictor.test(inductive), expected_ind.sigmoid()
        ))

        validation = inductive.clone()
        validation.split_type = "validation"
        validation.split_stage = "validation"
        validation.split_setting = "semi-inductive"
        self.assertTrue(torch.allclose(
            predictor.test(validation), expected_ind.sigmoid()
        ))

    def test_deal_rejects_missing_split_metadata_and_query_training_edges(self):
        with self.assertRaisesRegex(ValueError, "query_mask"):
            link_prediction.LinkPrediction(
                self._training_data(), model_type="deal"
            )

        data = self._deal_training_data()
        predictor = link_prediction.LinkPrediction(data, model_type="deal")
        with self.assertRaisesRegex(ValueError, "only seen nodes"):
            predictor.train(negative_edge_index=torch.tensor([[0], [4]]))

    def test_feature_cosine_scores_features_without_training(self):
        data = Data(
            x=torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
            ]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_label_index=torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]]),
        )
        predictor = link_prediction.LinkPrediction(
            data, model_type="feature-cosine"
        )

        probabilities = predictor.test(data)

        self.assertTrue(torch.allclose(
            probabilities, torch.tensor([1.0, 0.5, 0.0, 0.5])
        ))
        self.assertFalse(predictor.trainable)
        self.assertIsNone(predictor.optimizer)
        with self.assertRaisesRegex(RuntimeError, "not trainable"):
            predictor.train()

    def test_feature_cosine_is_symmetric_and_ignores_message_passing_edges(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="feature-cosine"
        )
        original = predictor.test(data)
        reversed_data = data.clone()
        reversed_data.edge_label_index = data.edge_label_index.flip(0)
        reversed_data.edge_index = torch.empty((2, 0), dtype=torch.long)

        reversed_scores = predictor.test(reversed_data)

        self.assertTrue(torch.allclose(original, reversed_scores))
        self.assertTrue(torch.isfinite(original).all())
        self.assertTrue(((original >= 0) & (original <= 1)).all())

    def test_mlp_decoder_is_symmetric_and_trainable(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="mlp",
            conv_layer_type="GCNConv",
            hidden_channels=4,
            out_channels=2,
            decoder_hidden_channels=5,
            decoder_dropout=0.0,
        )
        z = predictor.model.encode(data.x, data.edge_index)
        reversed_edges = data.edge_label_index.flip(0)

        scores = predictor.model.decode(z, data.edge_label_index)
        reversed_scores = predictor.model.decode(z, reversed_edges)
        decoder_before = [
            parameter.detach().clone()
            for parameter in predictor.model.edge_decoder.parameters()
        ]
        loss = predictor.train(data)

        self.assertTrue(torch.allclose(scores, reversed_scores))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(
                decoder_before, predictor.model.edge_decoder.parameters()
            )
        ))

    def test_mlp_decoder_validates_configuration(self):
        data = self._training_data()
        for model_type in ("mlp", "asymmetric-mlp"):
            with self.subTest(model_type=model_type):
                with self.assertRaises(ValueError):
                    link_prediction.LinkPrediction(
                        data,
                        model_type=model_type,
                        decoder_hidden_channels=0,
                    )
                with self.assertRaises(ValueError):
                    link_prediction.LinkPrediction(
                        data, model_type=model_type, decoder_dropout=1.0
                    )

    def test_asymmetric_mlp_uses_latent_sources_and_feature_targets(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric-mlp",
            conv_layer_type="GCNConv",
            hidden_channels=4,
            out_channels=2,
            decoder_hidden_channels=5,
            decoder_dropout=0.0,
        )
        z = predictor.model.encode(data.x, data.edge_index)
        edges = data.edge_label_index
        source = z[edges[0]]
        target = data.x_reduced[edges[1]]
        expected_features = torch.cat(
            [source * target, torch.abs(source - target)], dim=-1
        )

        scores = predictor.model.decode_edges(z, edges, data)

        self.assertTrue(torch.allclose(
            scores,
            predictor.model.edge_decoder(expected_features).squeeze(-1),
        ))
        self.assertEqual(scores.shape, data.edge_label.shape)

    def test_asymmetric_mlp_decoder_is_trainable_and_regularized(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric-mlp",
            hidden_channels=4,
            out_channels=2,
            decoder_hidden_channels=5,
            regularization_type="latent-l2",
            regularization_strength=0.1,
        )
        decoder_before = [
            parameter.detach().clone()
            for parameter in predictor.model.edge_decoder.parameters()
        ]

        loss = predictor.train(data)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(predictor.last_regularization_loss.item(), 0.0)
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(
                decoder_before, predictor.model.edge_decoder.parameters()
            )
        ))

    def test_variational_model_samples_trains_and_infers_deterministically(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="variational",
            conv_layer_type="GCNConv",
            hidden_channels=4,
            out_channels=2,
            kl_weight=0.5,
        )

        predictor.model.train()
        first_sample = predictor.model.encode(data.x, data.edge_index)
        second_sample = predictor.model.encode(data.x, data.edge_index)
        self.assertEqual(predictor.model.mu.shape, (data.num_nodes, 2))
        self.assertEqual(predictor.model.logstd.shape, (data.num_nodes, 2))
        self.assertFalse(torch.equal(first_sample, second_sample))

        loss = predictor.train(data)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(predictor.last_regularization_loss.item(), 0.0)
        self.assertTrue(torch.allclose(
            loss,
            predictor.last_reconstruction_loss + predictor.last_regularization_loss,
        ))

        predictor.model.eval()
        first_mean = predictor.model.encode(data.x, data.edge_index)
        second_mean = predictor.model.encode(data.x, data.edge_index)
        self.assertTrue(torch.equal(first_mean, second_mean))
        self.assertTrue(torch.equal(first_mean, predictor.model.mu))

    def test_variational_kl_weight_scales_regularization(self):
        data = self._training_data()
        model = link_prediction.VGAE_Model(
            3, 4, 2, conv_layer_type="GCNConv", kl_weight=1.0
        )
        model.encode(data.x, data.edge_index)
        base_loss = model.regularization_loss()
        model.kl_weight = 2.0
        self.assertTrue(torch.allclose(model.regularization_loss(), 2 * base_loss))
        with self.assertRaises(ValueError):
            link_prediction.VGAE_Model(
                3, 4, 2, conv_layer_type="GCNConv", kl_weight=-1
            )

    def test_new_baselines_support_all_convolution_types(self):
        data = self._training_data()
        for model_type in ("mlp", "asymmetric-mlp", "variational"):
            for conv_layer_type in (
                "GCNConv", "GraphConv", "SAGEConv", "GATv2Conv"
            ):
                with self.subTest(
                    model_type=model_type, conv_layer_type=conv_layer_type
                ):
                    predictor = link_prediction.LinkPrediction(
                        data,
                        model_type=model_type,
                        conv_layer_type=conv_layer_type,
                        hidden_channels=4,
                        out_channels=2,
                    )
                    predictor.model.eval()
                    encoded = predictor.model.encode(data.x, data.edge_index)
                    scores = predictor.model.decode_edges(
                        encoded, data.edge_label_index, data
                    )
                    self.assertEqual(encoded.shape, (data.num_nodes, 2))
                    self.assertEqual(scores.shape, data.edge_label.shape)

    def test_encoder_output_shapes_are_finite(self):
        data = self._training_data()
        for model_class in (link_prediction.GAE_Model, link_prediction.AGAE_Model):
            for layer_name in ("GCNConv", "GraphConv", "SAGEConv", "GATv2Conv"):
                with self.subTest(model=model_class.__name__, layer=layer_name):
                    model = model_class(3, 4, 2, conv_layer_type=layer_name)
                    encoded = model.encode(data.x, data.edge_index)
                    self.assertEqual(encoded.shape, (data.num_nodes, 2))
                    self.assertTrue(torch.isfinite(encoded).all())

    def test_symmetric_decoder_matches_dot_products_and_is_symmetric(self):
        model = link_prediction.GAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]])
        edges = torch.tensor([[0, 1, 2], [1, 0, 1]])

        scores = model.decode(z, edges)

        expected = torch.tensor([11.0, 11.0, 5.0])
        self.assertTrue(torch.equal(scores, expected))
        self.assertEqual(scores[0].item(), scores[1].item())

    def test_asymmetric_decoder_matches_source_target_dot_products(self):
        model = link_prediction.AGAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_reduced = torch.tensor([[2.0, 0.0], [0.0, 5.0]])
        edges = torch.tensor([[0, 1], [1, 0]])

        scores = model.decode(z, edges, x_reduced)

        self.assertTrue(torch.equal(scores, torch.tensor([10.0, 6.0])))
        self.assertNotEqual(scores[0].item(), scores[1].item())

    def test_available_agae_regularizers_come_from_registry(self):
        self.assertEqual(
            link_prediction.available_agae_regularizers(),
            (
                "none", "latent-l2", "mse-alignment",
                "cosine-alignment", "graph-smoothness",
            ),
        )

    def test_agae_regularizers_are_finite_and_nonnegative(self):
        data = self._deal_training_data()
        z = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.0],
            [1000.0, -1000.0],
        ])
        model = link_prediction.AGAE_Model(3, 4, 2)

        for name, regularizer in link_prediction._AGAE_REGULARIZERS.items():
            with self.subTest(regularizer=name):
                penalty = regularizer(model, data, z)
                self.assertEqual(penalty.ndim, 0)
                self.assertTrue(torch.isfinite(penalty))
                self.assertGreaterEqual(penalty.item(), 0.0)

    def test_agae_node_regularizers_exclude_query_nodes(self):
        data = self._deal_training_data()
        z = torch.ones((data.num_nodes, 2))
        baseline = {
            name: link_prediction._AGAE_REGULARIZERS[name](None, data, z)
            for name in ("latent-l2", "mse-alignment", "cosine-alignment")
        }
        z[4] = torch.tensor([1e6, -1e6])
        data.x_reduced[4] = torch.tensor([-1e6, 1e6])

        for name, expected in baseline.items():
            with self.subTest(regularizer=name):
                actual = link_prediction._AGAE_REGULARIZERS[name](None, data, z)
                self.assertTrue(torch.allclose(actual, expected))

    def test_agae_regularization_strength_scales_only_penalty(self):
        data = self._deal_training_data()
        model = link_prediction.AGAE_Model(
            3, 4, 2,
            regularization_type="latent-l2",
            regularization_strength=1.0,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        edges = torch.tensor([[0, 1, 0, 3], [1, 2, 2, 0]])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

        first_total, first_components = model.compute_training_loss(
            data, edges, labels, criterion
        )
        model.regularization_strength = 2.0
        second_total, second_components = model.compute_training_loss(
            data, edges, labels, criterion
        )

        self.assertTrue(torch.allclose(
            second_components["regularization"],
            2 * first_components["regularization"],
        ))
        self.assertTrue(torch.allclose(
            first_components["reconstruction"],
            second_components["reconstruction"],
        ))
        self.assertTrue(torch.allclose(
            first_total,
            first_components["reconstruction"]
            + first_components["regularization"],
        ))
        self.assertTrue(torch.allclose(
            second_total,
            second_components["reconstruction"]
            + second_components["regularization"],
        ))

    def test_agae_regularization_and_optimizer_configuration_are_validated(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric",
            learning_rate=0.005,
            weight_decay=0.01,
        )
        self.assertEqual(predictor.optimizer.param_groups[0]["lr"], 0.005)
        self.assertEqual(predictor.optimizer.param_groups[0]["weight_decay"], 0.01)

        invalid_configs = (
            {"regularization_type": "unknown"},
            {"regularization_strength": -1.0},
            {"learning_rate": 0.0},
            {"weight_decay": -1.0},
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    link_prediction.LinkPrediction(
                        data, model_type="asymmetric", **config
                    )

    def test_decode_all_applies_score_threshold(self):
        symmetric = link_prediction.GAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        asymmetric = link_prediction.AGAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        x_reduced = torch.tensor([[0.4, 0.0], [0.0, 0.6]])

        self.assertTrue(torch.equal(
            symmetric.decode_all(z), torch.tensor([[0, 1], [0, 1]])
        ))
        self.assertTrue(torch.equal(
            asymmetric.decode_all(z, x_reduced), torch.tensor([[1], [1]])
        ))

    def test_link_prediction_constructor_configuration(self):
        data = self._training_data()
        symmetric = link_prediction.LinkPrediction(
            data, model_type="symmetric", conv_layer_type="GraphConv",
            hidden_channels=7, out_channels=2,
        )
        asymmetric = link_prediction.LinkPrediction(
            data, model_type="asymmetric", conv_layer_type="SAGEConv",
            hidden_channels=6, out_channels=2,
        )

        self.assertIsInstance(symmetric.model, link_prediction.GAE_Model)
        self.assertIsInstance(symmetric.model.conv1, GraphConv)
        self.assertEqual(symmetric.model.conv1.out_channels, 7)
        self.assertEqual(symmetric.model.conv2.out_channels, 2)
        self.assertIsInstance(asymmetric.model, link_prediction.AGAE_Model)
        self.assertIsInstance(asymmetric.model.conv1, SAGEConv)
        self.assertEqual(asymmetric.model.conv1.out_channels, 6)
        self.assertIsInstance(symmetric.criterion, torch.nn.BCEWithLogitsLoss)
        optimizer_params = {id(param) for group in symmetric.optimizer.param_groups
                            for param in group["params"]}
        self.assertEqual(optimizer_params, {id(param) for param in symmetric.model.parameters()})

    def test_default_constructor_uses_symmetric_gcn(self):
        predictor = link_prediction.LinkPrediction(self._training_data())
        self.assertIsInstance(predictor.model, link_prediction.GAE_Model)
        self.assertIsInstance(predictor.model.conv1, GCNConv)

    def test_gatv2_options_are_forwarded_and_output_shape_is_preserved(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="asymmetric",
            conv_layer_type="GATv2Conv",
            hidden_channels=4,
            out_channels=2,
            heads=3,
            concat=True,
            dropout=0.2,
            share_weights=True,
        )

        self.assertIsInstance(predictor.model.conv1, GATv2Conv)
        self.assertEqual(predictor.model.conv1.heads, 3)
        self.assertTrue(predictor.model.conv1.concat)
        self.assertEqual(predictor.model.conv1.dropout, 0.2)
        self.assertTrue(predictor.model.conv1.share_weights)
        self.assertEqual(predictor.model.conv2.heads, 1)
        self.assertFalse(predictor.model.conv2.concat)
        encoded = predictor.model.encode(data.x, data.edge_index)
        self.assertEqual(encoded.shape, (data.num_nodes, 2))
        loss = predictor.train(data)
        probabilities = predictor.test(data)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(probabilities.shape, (data.edge_label_index.size(1),))

    def test_gatv2_options_can_be_passed_as_dictionary(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data,
            model_type="symmetric",
            conv_layer_type="GATv2Conv",
            hidden_channels=4,
            out_channels=2,
            gatv2_kwargs={"heads": 2, "concat": False, "dropout": 0.1},
        )

        self.assertEqual(predictor.model.conv1.heads, 2)
        self.assertFalse(predictor.model.conv1.concat)
        self.assertEqual(predictor.model.conv1.dropout, 0.1)

    def test_gatv2_kwargs_must_be_dictionary(self):
        with self.assertRaisesRegex(TypeError, "gatv2_kwargs"):
            link_prediction.LinkPrediction(
                self._training_data(),
                conv_layer_type="GATv2Conv",
                gatv2_kwargs="invalid",
            )

    @patch("link_prediction.negative_sampling")
    def test_asymmetric_training_step_updates_parameters(self, mock_sampling):
        data = self._training_data()
        mock_sampling.return_value = torch.tensor([[0, 4], [3, 1]])
        predictor = link_prediction.LinkPrediction(
            data, model_type="asymmetric", hidden_channels=4, out_channels=2
        )
        before = [parameter.detach().clone() for parameter in predictor.model.parameters()]

        loss = predictor.train(data)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            not torch.equal(old, new) for old, new in zip(before, predictor.model.parameters())
        ))
        self.assertTrue(torch.allclose(
            loss,
            predictor.last_reconstruction_loss + predictor.last_regularization_loss,
        ))
        self.assertEqual(predictor.last_regularization_loss.item(), 0.0)
        mock_sampling.assert_called_once_with(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1),
        )
        sampled = mock_sampling.return_value
        positives = {tuple(edge) for edge in data.edge_index.t().tolist()}
        self.assertEqual(sampled.size(1), data.edge_label_index.size(1))
        self.assertTrue(all(tuple(edge) not in positives for edge in sampled.t().tolist()))

    def test_train_without_argument_uses_constructor_data(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="asymmetric", hidden_channels=4, out_channels=2
        )
        loss = predictor.train()
        self.assertTrue(torch.isfinite(loss))

    def test_train_accepts_precomputed_negative_edges(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        negative_edges = torch.tensor([[0, 4], [3, 1]])

        with patch("link_prediction.negative_sampling") as mock_sampling:
            loss = predictor.train(
                data, negative_edge_index=negative_edges
            )

        self.assertTrue(torch.isfinite(loss))
        mock_sampling.assert_not_called()

    def test_train_validates_precomputed_negative_edges(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        invalid_edges = (
            torch.tensor([0, 1]),
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[0], [data.num_nodes]]),
        )
        for edge_index in invalid_edges:
            with self.subTest(edge_index=edge_index):
                with self.assertRaises((TypeError, ValueError)):
                    predictor.train(
                        data, negative_edge_index=edge_index
                    )

    def test_seeded_negative_sampling_is_reproducible_and_rng_safe(self):
        data = self._training_data()
        torch.manual_seed(19)
        expected_next_random = torch.rand(1)
        torch.manual_seed(19)
        random.seed(19)
        expected_next_python_random = random.random()
        random.seed(19)

        first = link_prediction.sample_training_negative_edges(data, seed=7)
        actual_next_random = torch.rand(1)
        actual_next_python_random = random.random()
        second = link_prediction.sample_training_negative_edges(data, seed=7)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(actual_next_random, expected_next_random))
        self.assertEqual(actual_next_python_random, expected_next_python_random)

    def test_training_negative_sampling_excludes_query_nodes(self):
        data = self._deal_training_data()

        negatives = link_prediction.sample_training_negative_edges(data, seed=7)

        self.assertTrue(bool(data.seen_node_mask[negatives].all()))

    def test_graph_context_model_can_override_training_loss(self):
        class ContextModel(link_prediction.LinkPredictionModel):
            def __init__(
                self, in_channels, hidden_channels, out_channels, train_data,
                **kwargs
            ):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.constructor_data = train_data
                self.received_edges = None
                self.received_labels = None

            def decode_edges(self, z, edge_label_index, data):
                raise AssertionError("custom training hook should be used")

            def compute_training_loss(
                self, data, edge_label_index, edge_label, criterion
            ):
                self.received_edges = edge_label_index.clone()
                self.received_labels = edge_label.clone()
                custom = self.weight.square()
                return custom, {"custom_alignment": custom}

        data = self._training_data()
        negative_edges = torch.tensor([[0, 4], [3, 1]])
        spec = link_prediction.ModelSpec(
            ContextModel,
            uses_conv_layer=False,
            uses_graph_context=True,
        )
        with patch.dict(
            link_prediction._MODEL_REGISTRY, {"context-test": spec}
        ):
            predictor = link_prediction.LinkPrediction(
                data,
                model_type="context-test",
                hidden_channels=4,
                out_channels=2,
            )
            loss = predictor.train(
                negative_edge_index=negative_edges
            )

        self.assertIs(predictor.model.constructor_data, data)
        self.assertEqual(
            set(predictor.last_loss_components), {"custom_alignment"}
        )
        self.assertTrue(torch.equal(
            predictor.model.received_edges,
            torch.cat([data.edge_label_index, negative_edges], dim=1),
        ))
        self.assertTrue(torch.equal(
            predictor.model.received_labels,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        ))
        self.assertTrue(torch.isfinite(loss))

    def test_symmetric_training_step_returns_finite_loss(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        loss = predictor.train(data)
        self.assertTrue(torch.isfinite(loss))

    def test_asymmetric_test_returns_probabilities_and_sets_eval_mode(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="asymmetric", hidden_channels=4, out_channels=2
        )
        probabilities = predictor.test(data)
        self.assertEqual(probabilities.shape, (data.edge_label_index.size(1),))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertFalse(predictor.model.training)

    def test_symmetric_test_returns_probabilities(self):
        data = self._training_data()
        predictor = link_prediction.LinkPrediction(
            data, model_type="symmetric", hidden_channels=4, out_channels=2
        )
        probabilities = predictor.test(data)
        self.assertEqual(probabilities.shape, (data.edge_label_index.size(1),))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_asymmetric_decoder_rejects_incompatible_feature_dimensions(self):
        model = link_prediction.AGAE_Model(2, 2, 2, conv_layer_type="GCNConv")
        z = torch.ones((2, 2))
        x_reduced = torch.ones((2, 3))
        with self.assertRaises(RuntimeError):
            model.decode(z, torch.tensor([[0], [1]]), x_reduced)

    def test_trainable_parameter_count_excludes_frozen_parameters(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2, bias=False),
        )
        model[1].weight.requires_grad_(False)

        self.assertEqual(
            link_prediction.count_trainable_parameters(model),
            3 * 4 + 4,
        )
        with self.assertRaises(TypeError):
            link_prediction.count_trainable_parameters(object())


class TestExperimentLogging(unittest.TestCase):
    @staticmethod
    def _result(**overrides):
        values = {
            "dataset": "Cora",
            "model_type": "asymmetric",
            "split_type": "inductive",
            "conv_layer_type": "GCNConv",
            "hidden_channels": 128,
            "out_channels": 64,
            "query_size": 10,
            "train_ratio": 0.8,
            "epochs": 100,
            "random_seed": 42,
            "training_time_seconds": 12.5,
            "trainable_parameter_count": 1234,
            "reducer_fit_time_seconds": 0.25,
            "roc_auc": 0.91,
            "average_precision": 0.87,
        }
        values.update(overrides)
        return ExperimentResult(**values)

    def test_logger_creates_file_and_round_trips_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.csv"
            run_id = CSVResultLogger(path).log(
                self._result(extra_fields={"learning_rate": 0.01})
            )

            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], run_id)
            self.assertTrue(rows[0]["timestamp_utc"].endswith("+00:00"))
            self.assertEqual(rows[0]["dataset"], "Cora")
            self.assertEqual(float(rows[0]["roc_auc"]), 0.91)
            self.assertEqual(float(rows[0]["training_time_seconds"]), 12.5)
            self.assertEqual(int(rows[0]["trainable_parameter_count"]), 1234)
            self.assertEqual(float(rows[0]["reducer_fit_time_seconds"]), 0.25)
            self.assertEqual(rows[0]["regularization_type"], "none")
            self.assertEqual(float(rows[0]["regularization_strength"]), 0.0)
            self.assertEqual(float(rows[0]["weight_decay"]), 0.0)
            self.assertEqual(rows[0]["normalize_features"], "True")
            self.assertEqual(json.loads(rows[0]["extra_fields"]), {"learning_rate": 0.01})

    def test_configuration_id_is_stable_and_order_independent(self):
        first = make_configuration_id({
            "dataset": "Cora", "model_type": "asymmetric", "epochs": 100,
        })
        second = make_configuration_id({
            "epochs": 100, "model_type": "asymmetric", "dataset": "Cora",
        })
        changed = make_configuration_id({
            "dataset": "Cora", "model_type": "asymmetric", "epochs": 200,
        })

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_result_validates_asymmetric_regularization_fields(self):
        result = self._result(
            regularization_type="cosine-alignment",
            regularization_strength=0.1,
            weight_decay=0.001,
        )
        self.assertEqual(result.regularization_type, "cosine-alignment")

        with self.assertRaisesRegex(ValueError, "only supported by asymmetric"):
            self._result(
                model_type="symmetric",
                regularization_type="latent-l2",
                regularization_strength=0.1,
            )

    def test_logger_round_trips_checkpoint_selection_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            CSVResultLogger(path).log(self._result(
                epochs_trained=50,
                best_epoch=37,
                selection_metric="roc_auc",
                best_validation_score=0.82,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                stopped_early=True,
            ))

            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))

        self.assertEqual(row["epochs_trained"], "50")
        self.assertEqual(row["best_epoch"], "37")
        self.assertEqual(row["selection_metric"], "roc_auc")
        self.assertEqual(float(row["best_validation_score"]), 0.82)
        self.assertEqual(row["early_stopping_patience"], "10")
        self.assertEqual(float(row["early_stopping_min_delta"]), 0.001)
        self.assertEqual(row["stopped_early"], "True")

    def test_logger_appends_without_repeating_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path)
            first_run_id = logger.log(self._result(split_type="inductive"))
            second_run_id = logger.log(
                self._result(split_type="transductive", query_size=0)
            )

            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

            self.assertEqual(rows[0], CSVResultLogger.FIELDNAMES)
            self.assertEqual(len(rows), 3)
            run_id_index = CSVResultLogger.FIELDNAMES.index("run_id")
            self.assertEqual(first_run_id, logger.run_id)
            self.assertEqual(second_run_id, logger.run_id)
            self.assertEqual(rows[1][run_id_index], logger.run_id)
            self.assertEqual(rows[2][run_id_index], logger.run_id)

    def test_separate_loggers_generate_different_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            first_logger = CSVResultLogger(path)
            second_logger = CSVResultLogger(path)

            first_logger.log(self._result())
            second_logger.log(self._result())

            self.assertNotEqual(first_logger.run_id, second_logger.run_id)

    def test_logger_accepts_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path, run_id="experiment-001")

            returned_run_id = logger.log(self._result())

            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(returned_run_id, "experiment-001")
            self.assertEqual(row["run_id"], "experiment-001")

    def test_logger_rejects_empty_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CSVResultLogger(Path(directory) / "results.csv", run_id=" ")

    def test_logger_can_reset_to_generated_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path)
            original_run_id = logger.log(self._result())

            reset_run_id = logger.reset_run_id()
            logged_run_id = logger.log(self._result())

            self.assertNotEqual(reset_run_id, original_run_id)
            self.assertEqual(logged_run_id, reset_run_id)

    def test_logger_can_reset_to_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            logger = CSVResultLogger(path)

            reset_run_id = logger.reset_run_id("experiment-002")
            logged_run_id = logger.log(self._result())

            self.assertEqual(reset_run_id, "experiment-002")
            self.assertEqual(logged_run_id, "experiment-002")
            with self.assertRaises(ValueError):
                logger.reset_run_id("")

    def test_logger_preserves_empty_optional_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            CSVResultLogger(path).log(self._result(roc_auc=None, average_precision=None))
            with path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["roc_auc"], "")
            self.assertEqual(row["average_precision"], "")

    def test_logger_rejects_mismatched_existing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                CSVResultLogger(path).log(self._result())

    def test_result_validates_parameters_and_metrics(self):
        invalid_values = [
            {"model_type": "unknown"},
            {"split_type": "unknown"},
            {"conv_layer_type": "unknown"},
            {"hidden_channels": 0},
            {"query_size": -1},
            {"normalize_features": "yes"},
            {"configuration_id": ""},
            {"train_ratio": 1.1},
            {"roc_auc": -0.1},
            {"loss": -1.0},
            {"training_time_seconds": -1.0},
            {"training_time_seconds": float("inf")},
            {"training_time_seconds": float("nan")},
            {"trainable_parameter_count": -1},
            {"trainable_parameter_count": 1.5},
            {"reducer_fit_time_seconds": -1.0},
            {"reducer_fit_time_seconds": float("inf")},
            {"epochs_trained": 101},
            {"epochs_trained": 10, "best_epoch": 11,
             "selection_metric": "roc_auc", "best_validation_score": 0.8},
            {"epochs_trained": 10, "best_epoch": 5},
            {"selection_metric": "accuracy"},
            {"best_validation_score": 1.1},
            {"early_stopping_patience": 0},
            {"early_stopping_min_delta": -0.1},
            {"stopped_early": "yes"},
            {"epochs_trained": 50, "stopped_early": True},
            {"epochs_trained": 100, "early_stopping_patience": 2,
             "stopped_early": True},
        ]
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self._result(**values)

    def test_logger_requires_experiment_result(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                CSVResultLogger(Path(directory) / "results.csv").log({})


class TestBenchmarkRunner(unittest.TestCase):
    @staticmethod
    def _write_config(directory, content):
        path = Path(directory) / "benchmark.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_and_expand_conditional_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, """
version: 1
output: out/test.csv
data_root: benchmark-data
seeds: [0, 1]
defaults:
  query_size: 5
  epochs: 2
  normalize_features: false
datasets: [Cora, CiteSeer]
models:
  - type: symmetric
    conv_layer_types: [GCNConv, SAGEConv]
  - type: asymmetric
    conv_layer_types: [GCNConv]
    reduction_methods: [pca, truncated-svd]
    regularization:
      - {type: none, strength: 0.0}
      - {type: cosine-alignment, strength: 0.01}
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(config.datasets, ("cora", "citeseer"))
        self.assertEqual(config.seeds, (0, 1))
        self.assertEqual(len(jobs), 12)
        self.assertEqual(
            benchmark.describe_benchmark(config, jobs),
            {
                "datasets": 2,
                "jobs": 12,
                "seeds_per_job": 2,
                "model_runs": 24,
                "expected_result_rows": 48,
            },
        )
        first_args = jobs[0].to_namespace()
        self.assertEqual(first_args.dataset, "cora")
        self.assertEqual(first_args.seed, [0, 1])
        self.assertEqual(first_args.query_size, 5)
        self.assertFalse(first_args.normalize_features)

    def test_selected_configuration_expands_dataset_specific_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            selected_path = Path(directory) / "selected.yaml"
            selected_path.write_text("""
version: 1
kind: selected-hyperparameters
selections:
  cora:
    asymmetric:
      hyperparameters:
        hidden_channels: 32
        out_channels: 16
        learning_rate: 0.005
        weight_decay: 0.001
        normalize_features: false
        conv_layer_type: SAGEConv
        reduction_method: truncated-svd
        regularization_type: cosine-alignment
        regularization_strength: 0.01
    deal:
      hyperparameters:
        hidden_channels: 64
        out_channels: 32
        learning_rate: 0.01
        weight_decay: 0.0001
        normalize_features: true
        conv_layer_type: GCNConv
        deal_profile: attribute-focused
        theta_ss: 0.1
        theta_aa: 1.0
        theta_sa: 0.5
        lambda_ss: 0.1
        lambda_aa: 1.0
        lambda_sa: 0.1
        similarity_scale: 1.0
        similarity_margin: 0.0
""", encoding="utf-8")
            path = self._write_config(directory, f"""
version: 2
kind: selected-hyperparameter-benchmark
output: out/preliminary.csv
seeds: [200, 201, 202]
selected_configurations:
  path: {selected_path}
  granularity: [dataset, architecture]
  require_all_pairs: true
defaults:
  query_fraction: 0.05
  validation_query_fraction: 0.05
  epochs: 100
datasets: [Cora]
models: [asymmetric, deal]
""")

            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(config.seeds, (200, 201, 202))
        asymmetric, deal = jobs
        self.assertEqual(asymmetric.conv_layer_type, "SAGEConv")
        self.assertEqual(asymmetric.reduction_method, "truncated-svd")
        self.assertEqual(asymmetric.regularization.type, "cosine-alignment")
        self.assertEqual(asymmetric.regularization.strength, 0.01)
        self.assertFalse(asymmetric.parameters["normalize_features"])
        self.assertEqual(asymmetric.parameters["hidden_channels"], 32)
        self.assertTrue(deal.parameters["normalize_features"])
        self.assertEqual(deal.parameters["deal_theta_ss"], 0.1)
        self.assertEqual(deal.parameters["deal_lambda_sa"], 0.1)
        self.assertEqual(
            benchmark.describe_benchmark(config, jobs)["expected_result_rows"],
            12,
        )

    def test_selected_configuration_requires_every_requested_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            selected_path = Path(directory) / "selected.yaml"
            selected_path.write_text("""
version: 1
kind: selected-hyperparameters
selections:
  cora: {}
""", encoding="utf-8")
            path = self._write_config(directory, f"""
version: 2
kind: selected-hyperparameter-benchmark
seeds: [200]
selected_configurations:
  path: {selected_path}
  granularity: [dataset, architecture]
  require_all_pairs: true
datasets: [Cora]
models: [symmetric]
""")
            config = benchmark.load_benchmark_config(path)

            with self.assertRaisesRegex(ValueError, "selections.cora.symmetric"):
                benchmark.expand_benchmark_config(config)

    def test_config_validation_rejects_incompatible_model_options(self):
        invalid_configs = (
            """
version: 1
seeds: [0]
datasets: [Cora]
models:
  - type: deal
    conv_layer_types: [GCNConv]
""",
            """
version: 1
seeds: [0]
datasets: [Cora]
models:
  - type: symmetric
    regularization:
      - {type: latent-l2, strength: 0.1}
""",
            """
version: 1
seeds: [0]
datasets: [unknown]
models:
  - type: symmetric
""",
            """
version: 1
seeds: [0]
datasets: [Cora]
models:
  - type: symmetric
    parameters:
      kl_weight: 0.5
""",
        )
        for content in invalid_configs:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = self._write_config(directory, content)
                with self.assertRaises(ValueError):
                    benchmark.load_benchmark_config(path)

    def test_expansion_rejects_invalid_resolved_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, """
version: 1
seeds: [0]
defaults:
  learning_rate: 0
datasets: [Cora]
models:
  - type: symmetric
""")
            config = benchmark.load_benchmark_config(path)

            with self.assertRaisesRegex(ValueError, "learning_rate"):
                benchmark.expand_benchmark_config(config)

    def test_asymmetric_mlp_grid_accepts_all_relevant_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, """
version: 1
seeds: [3]
datasets: [Cora]
models:
  - type: asymmetric-mlp
    conv_layer_types: [GCNConv]
    reduction_methods: [pca, truncated-svd]
    regularization:
      - {type: none, strength: 0.0}
      - {type: cosine-alignment, strength: 0.1}
    parameters:
      decoder_hidden_channels: 16
      decoder_dropout: 0.2
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(
            job.model_type == "asymmetric-mlp" for job in jobs
        ))
        self.assertEqual(
            {job.reduction_method for job in jobs}, {"pca", "truncated-svd"}
        )
        self.assertEqual(
            {job.regularization.type for job in jobs},
            {"none", "cosine-alignment"},
        )
        self.assertTrue(all(
            job.parameters["decoder_hidden_channels"] == 16 for job in jobs
        ))

    def test_dry_run_performs_no_execution_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            errors = Path(directory) / "errors.jsonl"
            path = self._write_config(directory, f"""
version: 1
output: {output}
error_output: {errors}
seeds: [0, 1]
datasets: [Cora]
models:
  - type: feature-cosine
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            runner = unittest.mock.Mock()

            summary = benchmark.execute_benchmark(
                config, jobs, dry_run=True, runner=runner, run_id="dry-run"
            )

            runner.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(errors.exists())
            self.assertEqual(summary.expected_result_rows, 4)

    def test_compact_progress_tracks_model_seed_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, f"""
version: 1
output: {Path(directory) / "results.csv"}
seeds: [3, 4]
datasets: [Cora]
models:
  - type: symmetric
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            received = []

            def runner(args, **kwargs):
                received.append((args, kwargs))
                callback = kwargs["progress_callback"]
                callback(
                    "run_started", dataset=args.dataset,
                    model_type="symmetric", seed=3, stage="training",
                )
                callback(
                    "run_completed", dataset=args.dataset,
                    model_type="symmetric", seed=3,
                )
                callback(
                    "run_started", dataset=args.dataset,
                    model_type="symmetric", seed=4, stage="training",
                )
                callback(
                    "run_skipped", dataset=args.dataset,
                    model_type="symmetric", seed=4,
                )

            progress = unittest.mock.Mock()
            with patch("benchmark.tqdm", return_value=progress) as mock_tqdm:
                summary = benchmark.execute_benchmark(
                    config, jobs, verbose=True, runner=runner
                )

        self.assertEqual(summary.jobs_succeeded, 1)
        self.assertTrue(received[0][0].verbose)
        self.assertTrue(received[0][1]["compact_progress"])
        self.assertEqual(progress.update.call_args_list, [
            unittest.mock.call(1), unittest.mock.call(1)
        ])
        mock_tqdm.assert_called_once()
        self.assertEqual(mock_tqdm.call_args.kwargs["total"], 2)
        self.assertEqual(mock_tqdm.call_args.kwargs["desc"], "Benchmark runs")

    def test_execution_shares_run_id_and_isolates_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = Path(directory) / "errors.jsonl"
            path = self._write_config(directory, f"""
version: 1
output: {Path(directory) / "results.csv"}
error_output: {errors}
seeds: [3]
datasets: [Cora]
models:
  - type: symmetric
  - type: feature-cosine
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            calls = []

            def runner(args, **kwargs):
                calls.append((args, kwargs))
                if args.model_type == ["symmetric"]:
                    raise RuntimeError("simulated failure")

            summary = benchmark.execute_benchmark(
                config, jobs, runner=runner, run_id="benchmark-001"
            )
            records = [
                json.loads(line)
                for line in errors.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary.jobs_succeeded, 1)
        self.assertEqual(summary.jobs_failed, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            {kwargs["run_id"] for _, kwargs in calls}, {"benchmark-001"}
        )
        self.assertIs(calls[0][1]["completed_results"], calls[1][1]["completed_results"])
        self.assertEqual(records[0]["error_type"], "RuntimeError")
        self.assertEqual(records[0]["message"], "simulated failure")

    def test_resume_loads_completed_configuration_seed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            row = {name: "" for name in CSVResultLogger.FIELDNAMES}
            row.update({
                "configuration_id": "config-1",
                "random_seed": "7",
            })
            with output.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=CSVResultLogger.FIELDNAMES
                )
                writer.writeheader()
                writer.writerow(row)
            path = self._write_config(directory, f"""
version: 1
output: {output}
seeds: [7]
datasets: [Cora]
models:
  - type: feature-cosine
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)
            received = []

            def runner(args, **kwargs):
                received.append(kwargs["completed_results"].copy())

            benchmark.execute_benchmark(
                config, jobs, resume=True, runner=runner
            )

        self.assertEqual(received, [{("config-1", 7)}])

    def test_fail_fast_records_and_reraises(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = Path(directory) / "errors.jsonl"
            path = self._write_config(directory, f"""
version: 1
error_output: {errors}
seeds: [0]
datasets: [Cora]
models:
  - type: symmetric
""")
            config = benchmark.load_benchmark_config(path)
            jobs = benchmark.expand_benchmark_config(config)

            with self.assertRaisesRegex(RuntimeError, "stop now"):
                benchmark.execute_benchmark(
                    config,
                    jobs,
                    fail_fast=True,
                    runner=unittest.mock.Mock(
                        side_effect=RuntimeError("stop now")
                    ),
                )
            self.assertTrue(errors.exists())


class TestSearchRunner(unittest.TestCase):
    @staticmethod
    def _graph(num_nodes=20):
        edges = [(index, (index + 1) % num_nodes) for index in range(num_nodes)]
        edges += [(index, (index + 4) % num_nodes) for index in range(num_nodes)]
        directed = edges + [(target, source) for source, target in edges]
        edge_index = torch.unique(
            torch.tensor(directed, dtype=torch.long).t(), dim=1
        )
        data = Data(
            x=torch.arange(num_nodes * 4, dtype=torch.float).reshape(num_nodes, 4) + 1,
            edge_index=edge_index,
        )
        data.dataset_name = "Toy"
        data.dataset_family = "Test"
        return data

    def test_pilot_spec_generates_deterministic_stratified_candidates(self):
        config = search.load_search_config("configs/pilot_search.yaml")
        first = search.generate_candidates(config)
        second = search.generate_candidates(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8 * 6 * 3)
        for dataset in config.datasets:
            for architecture in config.architectures:
                group = [
                    candidate for candidate in first
                    if candidate.dataset == dataset
                    and candidate.architecture == architecture
                ]
                self.assertEqual(len(group), 3)
                self.assertEqual(
                    {candidate.hyperparameters["normalize_features"] for candidate in group},
                    {True, False},
                )
                self.assertEqual(group[0].hyperparameters["hidden_channels"], 128)
                self.assertEqual(group[0].hyperparameters["learning_rate"], 0.01)
                if architecture in {"asymmetric", "asymmetric-mlp"}:
                    self.assertEqual(
                        {candidate.hyperparameters["reduction_method"] for candidate in group},
                        {"pca", "truncated-svd", "sparse-random-projection"},
                    )
                    self.assertEqual(
                        {candidate.hyperparameters["regularization_strength"] for candidate in group},
                        {0.0, 0.001, 0.01},
                    )
                if architecture == "variational":
                    self.assertEqual(
                        {candidate.hyperparameters["kl_weight"] for candidate in group},
                        {0.01, 0.1, 1.0},
                    )
                if architecture == "deal":
                    self.assertEqual(
                        {candidate.hyperparameters["deal_profile"] for candidate in group},
                        {"balanced", "attribute-focused", "alignment-focused"},
                    )

    def test_validation_data_is_isolated_and_does_not_expose_test_splits(self):
        source = self._graph()
        original_features = source.x.clone()
        splits = utils.generate_comparable_splits(
            source,
            query_fraction=0.05,
            validation_query_fraction=0.05,
            seed=4,
        )
        with patch(
            "experiment_pipeline.calculate_reduced_features",
            wraps=utils.calculate_reduced_features,
        ) as reducer:
            prepared = experiment_pipeline.prepare_validation_data(
                source,
                splits,
                model_type="asymmetric",
                normalize_features=True,
                out_channels=2,
                reduction_method="pca",
                random_state=4,
            )

        self.assertFalse(hasattr(prepared, "test_transductive"))
        self.assertFalse(hasattr(prepared, "test_inductive"))
        self.assertTrue(torch.equal(source.x, original_features))
        self.assertTrue(torch.equal(prepared.train.edge_index, splits.train.edge_index))
        self.assertTrue(torch.equal(
            reducer.call_args.kwargs["fit_nodes"], splits.train.visible_node_mask
        ))
        self.assertEqual(prepared.train.x_reduced.shape, (source.num_nodes, 2))
        self.assertIsNotNone(prepared.reducer_fit_time_seconds)

    def test_selection_uses_tolerance_then_ap_then_parameter_count(self):
        base = search.load_search_config("configs/pilot_search.yaml")
        candidates = [
            candidate for candidate in search.generate_candidates(base)
            if candidate.dataset == "cora" and candidate.architecture == "symmetric"
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base,
                datasets=("cora",),
                architectures=("symmetric",),
                history_path=Path(directory) / "history.csv",
                failures_path=Path(directory) / "errors.jsonl",
                aggregate_path=Path(directory) / "aggregates.csv",
                selected_path=Path(directory) / "selected.yaml",
            )
            scores = ((0.9000, 0.70, 100), (0.8995, 0.80, 200), (0.80, 0.90, 50))
            rows = []
            for candidate, (roc_auc, average_precision, parameters) in zip(
                candidates, scores
            ):
                for seed in config.tuning_seeds:
                    rows.append({
                        "candidate_id": candidate.candidate_id,
                        "tuning_seed": str(seed),
                        "status": "success",
                        "validation_roc_auc": str(roc_auc),
                        "validation_average_precision": str(average_precision),
                        "trainable_parameter_count": str(parameters),
                    })

            winners = search.aggregate_and_select(
                config, candidates, rows, "search-1"
            )
            selected = yaml.safe_load(
                config.selected_path.read_text(encoding="utf-8")
            )

        winner = winners[("cora", "symmetric")]
        self.assertEqual(winner["candidate_id"], candidates[1].candidate_id)
        self.assertEqual(
            selected["selections"]["cora"]["symmetric"]["candidate_id"],
            candidates[1].candidate_id,
        )

    @patch("search.load_dataset")
    def test_toy_search_logs_selects_and_resumes(self, mock_load):
        mock_load.return_value = self._graph()
        base = search.load_search_config("configs/pilot_search.yaml")
        raw = copy.deepcopy(base.raw)
        raw["training"]["max_epochs"] = 1
        raw["training"]["early_stopping"]["patience"] = 1
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base,
                raw=raw,
                datasets=("cora",),
                architectures=("symmetric",),
                tuning_seeds=(1,),
                history_path=Path(directory) / "history.csv",
                failures_path=Path(directory) / "errors.jsonl",
                aggregate_path=Path(directory) / "aggregates.csv",
                selected_path=Path(directory) / "selected.yaml",
            )
            candidate = search.SearchCandidate(
                "toy-candidate", "cora", "symmetric",
                {
                    "hidden_channels": 4,
                    "out_channels": 2,
                    "learning_rate": 0.01,
                    "weight_decay": 0.001,
                    "normalize_features": False,
                    "conv_layer_type": "GCNConv",
                },
            )
            first = search.execute_search(
                config, (candidate,), run_id="pilot-1"
            )
            second = search.execute_search(
                config, (candidate,), resume=True
            )
            with config.history_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(first.succeeded, 1)
        self.assertEqual(first.selected, 1)
        self.assertEqual(second.skipped, 1)
        self.assertEqual(second.run_id, "pilot-1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], "toy-candidate")

    @patch("search.generate_comparable_splits")
    @patch("search.load_dataset")
    def test_shared_split_failure_is_cached_and_logged_once(
        self, mock_load, mock_split
    ):
        mock_load.return_value = self._graph()
        mock_split.side_effect = ValueError("infeasible shared split")
        base = search.load_search_config("configs/pilot_search.yaml")
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base,
                datasets=("cora",),
                architectures=("symmetric",),
                tuning_seeds=(1,),
                history_path=Path(directory) / "history.csv",
                failures_path=Path(directory) / "errors.jsonl",
                aggregate_path=Path(directory) / "aggregates.csv",
                selected_path=Path(directory) / "selected.yaml",
            )
            candidates = (
                search.SearchCandidate(
                    "candidate-1", "cora", "symmetric",
                    {
                        "hidden_channels": 4,
                        "out_channels": 2,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "normalize_features": False,
                        "conv_layer_type": "GCNConv",
                    },
                ),
                search.SearchCandidate(
                    "candidate-2", "cora", "symmetric",
                    {
                        "hidden_channels": 8,
                        "out_channels": 2,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "normalize_features": False,
                        "conv_layer_type": "GCNConv",
                    },
                ),
            )

            summary = search.execute_search(
                config, candidates, run_id="shared-failure"
            )
            records = [
                json.loads(line)
                for line in config.failures_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary.total_runs, 2)
        self.assertEqual(summary.failed, 2)
        self.assertEqual(summary.succeeded, 0)
        self.assertEqual(mock_split.call_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scope"], "dataset-seed")
        self.assertEqual(records[0]["stage"], "data-splitting")
        self.assertEqual(records[0]["dataset"], "cora")
        self.assertEqual(records[0]["tuning_seed"], 1)
        self.assertEqual(records[0]["affected_runs"], 2)


class TestDatasetLoading(unittest.TestCase):
    @staticmethod
    def _graph():
        return Data(
            x=torch.tensor([[2.0, 2.0], [1.0, 3.0], [4.0, 0.0]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )

    def test_available_datasets_has_stable_names(self):
        self.assertEqual(datasets.available_datasets(), [
            "amazon-computers", "amazon-photo", "citeseer", "coauthor-cs",
            "cora", "dblp", "deezer-europe", "pubmed", "wikics"
        ])

    @patch("datasets.Planetoid")
    def test_planetoid_names_dispatch_to_canonical_names(self, mock_planetoid):
        expected_names = {
            "cora": "Cora",
            "CITESEER": "CiteSeer",
            "PubMed": "PubMed",
        }
        for supplied_name, canonical_name in expected_names.items():
            with self.subTest(name=supplied_name):
                mock_planetoid.reset_mock()
                mock_planetoid.return_value.__getitem__.return_value = self._graph()

                data = datasets.load_dataset(
                    supplied_name, root=Path("benchmark-data"), normalize_features=False
                )

                mock_planetoid.assert_called_once_with(
                    root="benchmark-data", name=canonical_name
                )
                self.assertEqual(data.dataset_name, canonical_name)
                self.assertEqual(data.dataset_family, "Planetoid")

    @patch("datasets.Amazon")
    def test_amazon_photo_loader_and_aliases(self, mock_amazon):
        mock_amazon.return_value.__getitem__.return_value = self._graph()
        for supplied_name in ("amazon-photo", "Amazon Photo", "amazon_photo"):
            with self.subTest(name=supplied_name):
                mock_amazon.reset_mock()
                data = datasets.load_dataset(
                    supplied_name, root="benchmark-data", normalize_features=False
                )
                mock_amazon.assert_called_once_with(
                    root="benchmark-data", name="Photo"
                )
                self.assertEqual(data.dataset_name, "AmazonPhoto")
                self.assertEqual(data.dataset_family, "Amazon")

    @patch("datasets.Amazon")
    def test_amazon_computers_loader_and_aliases(self, mock_amazon):
        mock_amazon.return_value.__getitem__.return_value = self._graph()
        for supplied_name in (
            "amazon-computers", "Amazon Computers", "amazon_computers"
        ):
            with self.subTest(name=supplied_name):
                mock_amazon.reset_mock()
                data = datasets.load_dataset(
                    supplied_name, root="benchmark-data", normalize_features=False
                )
                mock_amazon.assert_called_once_with(
                    root="benchmark-data", name="Computers"
                )
                self.assertEqual(data.dataset_name, "AmazonComputers")
                self.assertEqual(data.dataset_family, "Amazon")

    @patch("datasets.Coauthor")
    def test_coauthor_cs_loader_and_aliases(self, mock_coauthor):
        mock_coauthor.return_value.__getitem__.return_value = self._graph()
        for supplied_name in ("coauthor-cs", "Coauthor CS", "coauthor_cs"):
            with self.subTest(name=supplied_name):
                mock_coauthor.reset_mock()
                data = datasets.load_dataset(
                    supplied_name, root="benchmark-data", normalize_features=False
                )
                mock_coauthor.assert_called_once_with(root="benchmark-data", name="CS")
                self.assertEqual(data.dataset_name, "CoauthorCS")
                self.assertEqual(data.dataset_family, "Coauthor")

    @patch("datasets.CitationFull")
    def test_dblp_loader(self, mock_citation_full):
        mock_citation_full.return_value.__getitem__.return_value = self._graph()
        data = datasets.load_dataset(
            "DBLP", root="benchmark-data", normalize_features=False
        )
        mock_citation_full.assert_called_once_with(
            root="benchmark-data", name="DBLP", to_undirected=True
        )
        self.assertEqual(data.dataset_name, "DBLP")
        self.assertEqual(data.dataset_family, "CitationFull")

    @patch("datasets.WikiCS")
    def test_wikics_loader_requests_undirected_graph(self, mock_wikics):
        mock_wikics.return_value.__getitem__.return_value = self._graph()
        data = datasets.load_dataset(
            "wiki-cs", root="benchmark-data", normalize_features=False
        )
        mock_wikics.assert_called_once_with(
            root="benchmark-data", is_undirected=True
        )
        self.assertEqual(data.dataset_name, "WikiCS")
        self.assertEqual(data.dataset_family, "WikiCS")

    @patch("datasets.DeezerEurope")
    def test_deezer_loader_and_alias(self, mock_deezer):
        mock_deezer.return_value.__getitem__.return_value = self._graph()
        data = datasets.load_dataset(
            "Deezer Europe", root="benchmark-data", normalize_features=False
        )
        mock_deezer.assert_called_once_with(root="benchmark-data")
        self.assertEqual(data.dataset_name, "DeezerEurope")
        self.assertEqual(data.dataset_family, "Deezer")

    @patch("datasets.Planetoid")
    def test_loader_clones_normalizes_and_moves_graph(self, mock_planetoid):
        original = self._graph()
        mock_planetoid.return_value.__getitem__.return_value = original

        loaded = datasets.load_dataset("cora", device=torch.device("cpu"))

        self.assertIsNot(loaded, original)
        self.assertTrue(torch.allclose(loaded.x.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.equal(original.x, self._graph().x))
        self.assertEqual(loaded.x.device.type, "cpu")

    @patch("datasets.Planetoid")
    def test_loader_can_skip_feature_normalization(self, mock_planetoid):
        original = self._graph()
        mock_planetoid.return_value.__getitem__.return_value = original

        loaded = datasets.load_dataset("cora", normalize_features=False)

        self.assertTrue(torch.equal(loaded.x, original.x))

    @patch("datasets.Planetoid")
    def test_loader_removes_self_loops_without_mutating_source(
        self, mock_planetoid
    ):
        original = self._graph()
        original.edge_index = torch.tensor([
            [0, 0, 1, 1, 1, 2],
            [0, 1, 0, 1, 2, 1],
        ])
        original.edge_attr = torch.arange(6, dtype=torch.float).unsqueeze(1)
        original_edges = original.edge_index.clone()
        original_edge_attr = original.edge_attr.clone()
        mock_planetoid.return_value.__getitem__.return_value = original

        loaded = datasets.load_dataset("cora", normalize_features=False)

        self.assertFalse(torch.any(loaded.edge_index[0] == loaded.edge_index[1]))
        self.assertTrue(torch.equal(
            loaded.edge_index,
            torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ))
        self.assertTrue(torch.equal(
            loaded.edge_attr,
            torch.tensor([[1.0], [2.0], [4.0], [5.0]]),
        ))
        self.assertTrue(torch.equal(original.edge_index, original_edges))
        self.assertTrue(torch.equal(original.edge_attr, original_edge_attr))

    def test_loader_rejects_unknown_name_and_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "Available datasets"):
            datasets.load_dataset("unknown")
        with self.assertRaises(TypeError):
            datasets.load_dataset("cora", root=123)
        with self.assertRaises(TypeError):
            datasets.load_dataset("cora", normalize_features="yes")
        with self.assertRaises(TypeError):
            datasets.load_dataset("cora", device=3)
        with self.assertRaisesRegex(ValueError, "Invalid device"):
            datasets.load_dataset("cora", device="not-a-device")

    @patch("datasets.Planetoid")
    def test_loader_rejects_missing_features(self, mock_planetoid):
        graph = self._graph()
        graph.x = None
        mock_planetoid.return_value.__getitem__.return_value = graph
        with self.assertRaisesRegex(ValueError, "node features"):
            datasets.load_dataset("cora")

    @patch("datasets.Planetoid")
    def test_loader_rejects_directed_graphs(self, mock_planetoid):
        graph = self._graph()
        graph.edge_index = torch.tensor([[0, 1], [1, 2]])
        mock_planetoid.return_value.__getitem__.return_value = graph
        with self.assertRaisesRegex(ValueError, "undirected"):
            datasets.load_dataset("cora")


class TestMainPipeline(unittest.TestCase):
    @staticmethod
    def _graph():
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        directed_edges = edges + [(target, source) for source, target in edges]
        data = Data(
            x=torch.rand(8, 4),
            edge_index=torch.tensor(directed_edges).t().contiguous(),
        )
        data.dataset_name = "Toy"
        data.dataset_family = "Test"
        return data

    def test_seed_resolution_supports_repeated_and_generated_seeds(self):
        self.assertEqual(
            main._resolve_seeds(SimpleNamespace(seed=[3, 7], num_seeds=None)),
            (3, 7),
        )
        self.assertEqual(
            main._resolve_seeds(SimpleNamespace(seed=None, num_seeds=3)),
            (0, 1, 2),
        )
        self.assertEqual(
            main._resolve_seeds(SimpleNamespace(seed=None, num_seeds=None)),
            (42,),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            main._resolve_seeds(SimpleNamespace(seed=[3, 3], num_seeds=None))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            main._resolve_seeds(SimpleNamespace(seed=[3], num_seeds=2))

    def test_gatv2_cli_options_require_a_json_object(self):
        self.assertEqual(
            main._json_object('{"heads": 2, "dropout": 0.1}'),
            {"heads": 2, "dropout": 0.1},
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            main._json_object("not-json")
        with self.assertRaises(argparse.ArgumentTypeError):
            main._json_object("[1, 2]")

    @patch("main.load_dataset")
    def test_multi_seed_run_preserves_pairing_and_configuration_ids(
        self, mock_load
    ):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric", "feature-cosine"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=[1, 2],
                num_seeds=None,
                device="cpu",
                normalize_features=False,
            )
            with patch(
                "main.generate_comparable_splits",
                wraps=utils.generate_comparable_splits,
            ) as mock_splits:
                run_id = main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(mock_splits.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in mock_splits.call_args_list],
            [1, 2],
        )
        query_partitions = [
            (tuple(call.kwargs["validation_query"]), tuple(call.kwargs["query"]))
            for call in mock_splits.call_args_list
        ]
        self.assertNotEqual(query_partitions[0], query_partitions[1])
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["run_id"] for row in rows}, {run_id})
        self.assertEqual({row["random_seed"] for row in rows}, {"1", "2"})
        self.assertEqual({row["normalize_features"] for row in rows}, {"False"})

        grouped = {}
        for row in rows:
            key = (row["model_type"], row["split_type"])
            grouped.setdefault(key, []).append(row)
        self.assertEqual(len(grouped), 4)
        for repetitions in grouped.values():
            self.assertEqual(len(repetitions), 2)
            self.assertEqual(
                len({row["configuration_id"] for row in repetitions}), 1
            )
            self.assertEqual(
                {row["random_seed"] for row in repetitions}, {"1", "2"}
            )

    @patch("main.load_dataset")
    def test_completed_results_skip_model_evaluation_and_duplicate_rows(
        self, mock_load
    ):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["feature-cosine"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=5,
                device="cpu",
                normalize_features=False,
            )
            main.run_experiment(args, run_id="first")
            with output.open(newline="", encoding="utf-8") as csv_file:
                original_rows = list(csv.DictReader(csv_file))
            completed = {
                (row["configuration_id"], int(row["random_seed"]))
                for row in original_rows
            }

            with (
                patch("main._train_with_checkpoint_selection") as mock_train,
                patch("main.calculate_metrics") as mock_metrics,
            ):
                main.run_experiment(
                    args,
                    run_id="resume",
                    completed_results=completed,
                )
            with output.open(newline="", encoding="utf-8") as csv_file:
                resumed_rows = list(csv.DictReader(csv_file))

        mock_train.assert_not_called()
        mock_metrics.assert_not_called()
        self.assertEqual(resumed_rows, original_rows)

    @patch("main.load_dataset")
    def test_checkpoint_selection_restores_best_epoch_before_testing(
        self, mock_load
    ):
        mock_load.return_value = self._graph()
        evaluation_calls = []

        class FakePredictor:
            def __init__(self, train_data, **kwargs):
                self.trainable = True
                self.model_spec = SimpleNamespace(uses_conv_layer=True)
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                epoch = int(self.model.weight.item())
                evaluation_calls.append((data.split_stage, epoch))
                correct = torch.where(
                    data.edge_label == 1,
                    torch.tensor(0.9, device=data.edge_label.device),
                    torch.tensor(0.1, device=data.edge_label.device),
                )
                if data.split_stage == "validation" and epoch != 2:
                    return 1.0 - correct
                return correct

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=3,
                selection_metric="roc_auc",
                seed=42,
                device="cpu",
                normalize_features=False,
            )
            with patch("main.LinkPrediction", FakePredictor):
                main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(evaluation_calls, [
            ("validation", 1),
            ("validation", 2),
            ("validation", 3),
            ("test", 2),
            ("test", 2),
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["epochs_trained"] for row in rows}, {"3"})
        self.assertEqual({row["best_epoch"] for row in rows}, {"2"})
        self.assertEqual({row["selection_metric"] for row in rows}, {"roc_auc"})
        self.assertEqual(
            {float(row["best_validation_score"]) for row in rows}, {1.0}
        )
        self.assertEqual(
            {row["early_stopping_patience"] for row in rows}, {""}
        )
        self.assertEqual(
            {float(row["early_stopping_min_delta"]) for row in rows}, {0.0}
        )
        self.assertEqual({row["stopped_early"] for row in rows}, {"False"})

    def test_early_stopping_honors_patience_and_restores_best_state(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        train_data = Data(edge_index=torch.empty((2, 0), dtype=torch.long))
        validation_data = Data(edge_label=torch.tensor([1.0, 0.0]))
        validation_scores = [0.50, 0.70, 0.65, 0.60, 0.90]

        with (
            patch(
                "main.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ) as mock_negatives,
            patch(
                "main.calculate_metrics",
                side_effect=[
                    {"roc_auc": score} for score in validation_scores
                ],
            ) as mock_metrics,
        ):
            summary = main._train_with_checkpoint_selection(
                predictor,
                train_data,
                validation_data,
                epochs=5,
                seed=4,
                selection_metric="roc_auc",
                device="cpu",
                early_stopping_patience=2,
            )

        self.assertEqual(summary.epochs_trained, 4)
        self.assertEqual(summary.best_epoch, 2)
        self.assertEqual(summary.best_validation_score, 0.70)
        self.assertTrue(summary.stopped_early)
        self.assertEqual(predictor.model.weight.item(), 2.0)
        self.assertEqual(mock_negatives.call_count, 4)
        self.assertEqual(mock_metrics.call_count, 4)

    def test_early_stopping_min_delta_ignores_small_improvements(self):
        class FakePredictor:
            trainable = True

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.zero_()

            def train(self, negative_edge_index=None):
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return self.model.weight.detach().reshape(()).clone()

            def test(self, data):
                return torch.tensor([0.5, 0.5])

        predictor = FakePredictor()
        with (
            patch(
                "main.sample_training_negative_edges",
                return_value=torch.empty((2, 0), dtype=torch.long),
            ),
            patch(
                "main.calculate_metrics",
                side_effect=[{"roc_auc": 0.50}, {"roc_auc": 0.55}],
            ),
        ):
            summary = main._train_with_checkpoint_selection(
                predictor,
                Data(edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(edge_label=torch.tensor([1.0, 0.0])),
                epochs=5,
                seed=4,
                selection_metric="roc_auc",
                device="cpu",
                early_stopping_patience=1,
                early_stopping_min_delta=0.10,
            )

        self.assertEqual(summary.epochs_trained, 2)
        self.assertEqual(summary.best_epoch, 2)
        self.assertEqual(summary.best_validation_score, 0.55)
        self.assertEqual(predictor.model.weight.item(), 2.0)
        self.assertTrue(summary.stopped_early)

    @patch(
        "main.sample_training_negative_edges",
        wraps=link_prediction.sample_training_negative_edges,
    )
    @patch("main.load_dataset")
    def test_unspecified_model_type_runs_both_models_on_both_splits(
        self, mock_load, mock_negative_sampling
    ):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=None,
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            run_id = main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
        combinations = {(row["model_type"], row["split_type"]) for row in rows}
        self.assertEqual(combinations, {
            ("symmetric", "transductive"),
            ("symmetric", "inductive"),
            ("asymmetric", "transductive"),
            ("asymmetric", "inductive"),
            ("mlp", "transductive"),
            ("mlp", "inductive"),
            ("asymmetric-mlp", "transductive"),
            ("asymmetric-mlp", "inductive"),
            ("variational", "transductive"),
            ("variational", "inductive"),
            ("deal", "transductive"),
            ("deal", "inductive"),
            ("feature-cosine", "transductive"),
            ("feature-cosine", "inductive"),
        })
        self.assertEqual({row["run_id"] for row in rows}, {run_id})
        self.assertTrue(all(
            int(row["trainable_parameter_count"]) >= 0 for row in rows
        ))
        for model_type in {row["model_type"] for row in rows}:
            model_rows = [row for row in rows if row["model_type"] == model_type]
            self.assertEqual(
                len({row["trainable_parameter_count"] for row in model_rows}),
                1,
            )
        reduced_rows = [
            row for row in rows
            if row["model_type"] in {"asymmetric", "asymmetric-mlp"}
        ]
        self.assertTrue(all(
            float(row["reducer_fit_time_seconds"]) >= 0
            for row in reduced_rows
        ))
        self.assertEqual(
            len({row["reducer_fit_time_seconds"] for row in reduced_rows}),
            1,
        )
        self.assertTrue(all(
            row["reducer_fit_time_seconds"] == ""
            for row in rows
            if row["model_type"] not in {"asymmetric", "asymmetric-mlp"}
        ))
        trainable_rows = [
            row for row in rows if row["model_type"] != "feature-cosine"
        ]
        self.assertTrue(all(
            float(row["training_time_seconds"]) >= 0
            for row in trainable_rows
        ))
        self.assertEqual(
            {row["epochs_trained"] for row in trainable_rows}, {"1"}
        )
        self.assertEqual({row["best_epoch"] for row in trainable_rows}, {"1"})
        self.assertEqual(
            {row["selection_metric"] for row in trainable_rows}, {"roc_auc"}
        )
        for model_type in {
            row["model_type"] for row in trainable_rows
        }:
            model_runtimes = {
                row["training_time_seconds"]
                for row in trainable_rows
                if row["model_type"] == model_type
            }
            self.assertEqual(len(model_runtimes), 1)
        self.assertEqual(mock_negative_sampling.call_count, 6)
        self.assertEqual(
            [call.kwargs["seed"] for call in mock_negative_sampling.call_args_list],
            [args.seed] * 6,
        )
        cosine_rows = [
            row for row in rows if row["model_type"] == "feature-cosine"
        ]
        self.assertEqual(
            {row["trainable_parameter_count"] for row in cosine_rows}, {"0"}
        )
        self.assertEqual({row["conv_layer_type"] for row in cosine_rows}, {"None"})
        self.assertEqual({row["loss"] for row in cosine_rows}, {""})
        self.assertEqual(
            {row["training_time_seconds"] for row in cosine_rows}, {""}
        )
        self.assertEqual({row["epochs_trained"] for row in cosine_rows}, {"0"})
        self.assertEqual({row["best_epoch"] for row in cosine_rows}, {""})
        self.assertEqual({row["selection_metric"] for row in cosine_rows}, {""})
        self.assertEqual(
            {row["best_validation_score"] for row in cosine_rows}, {""}
        )
        self.assertEqual(
            {row["early_stopping_patience"] for row in cosine_rows}, {""}
        )
        self.assertEqual({row["stopped_early"] for row in cosine_rows}, {"False"})
        common_extra_fields = {
            "reduction_method", "validation_query_size", "learning_rate",
            "device", "query_selection", "validation_query_selection",
        }
        for row in rows:
            self.assertTrue(common_extra_fields.issubset(
                json.loads(row["extra_fields"])
            ))
        for model_type in ("mlp", "asymmetric-mlp"):
            mlp_config = json.loads(next(
                row["extra_fields"]
                for row in rows if row["model_type"] == model_type
            ))
            self.assertIn("decoder_hidden_channels", mlp_config)
            self.assertIn("decoder_dropout", mlp_config)
        variational_config = json.loads(next(
            row["extra_fields"]
            for row in rows if row["model_type"] == "variational"
        ))
        self.assertIn("kl_weight", variational_config)
        deal_rows = [row for row in rows if row["model_type"] == "deal"]
        self.assertEqual({row["conv_layer_type"] for row in deal_rows}, {"None"})
        deal_config = json.loads(deal_rows[0]["extra_fields"])
        self.assertEqual(deal_config["deal_theta_ss"], 1.0)
        self.assertEqual(deal_config["deal_lambda_sa"], 1.0)

    @patch("main.load_dataset")
    def test_explicit_model_type_list_runs_only_selected_models(self, mock_load):
        graph = self._graph()
        mock_load.return_value = graph
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["asymmetric", "asymmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            with patch(
                "experiment_pipeline.calculate_reduced_features",
                wraps=utils.calculate_reduced_features,
            ) as mock_reduction:
                main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["model_type"] for row in rows}, {"asymmetric"})
        query_nodes = utils.select_query_node_sets(
            graph, query_size=args.query_size, seed=args.seed
        )
        fit_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        fit_mask[list(query_nodes.validation)] = False
        fit_mask[list(query_nodes.test)] = False
        expected_mean = graph.x[fit_mask].mean(dim=0).numpy()
        reduced_graph = mock_reduction.call_args.args[0]
        self.assertTrue((reduced_graph.feature_reducer.mean_ == expected_mean).all())
        for row in rows:
            extra_fields = json.loads(row["extra_fields"])
            self.assertEqual(
                extra_fields["reduction_method"], "pca"
            )
            self.assertEqual(
                extra_fields["validation_query_size"], 1
            )
            self.assertEqual(extra_fields["learning_rate"], 0.01)
            self.assertEqual(extra_fields["device"], "cpu")
            self.assertEqual(extra_fields["query_selection"], {
                "mode": "size", "value": 1, "resolved_size": 1,
            })
            self.assertEqual(extra_fields["validation_query_selection"], {
                "mode": "match-test", "value": None, "resolved_size": 1,
            })

    @patch("main.load_dataset")
    def test_gatv2_kwargs_are_passed_to_model_and_logged(self, mock_load):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GATv2Conv",
                gatv2_kwargs={"heads": 2, "concat": False, "dropout": 0.1},
                hidden_channels=4,
                out_channels=2,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )
            with patch(
                "main.LinkPrediction", wraps=link_prediction.LinkPrediction
            ) as mock_predictor:
                main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(
            mock_predictor.call_args.kwargs["gatv2_kwargs"],
            args.gatv2_kwargs,
        )
        for row in rows:
            self.assertEqual(
                json.loads(row["extra_fields"])["gatv2_kwargs"],
                args.gatv2_kwargs,
            )

    @patch("main.load_dataset")
    def test_weight_decay_is_applied_to_non_asymmetric_trainable_models(
        self, mock_load
    ):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                learning_rate=0.01,
                weight_decay=0.001,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            main.run_experiment(args)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual({float(row["weight_decay"]) for row in rows}, {0.001})

    @patch("main.load_dataset")
    def test_compact_progress_emits_model_seed_events(self, mock_load):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(Path(directory) / "results.csv"),
                data_root="unused",
                model_type=["symmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                learning_rate=0.01,
                weight_decay=0.0,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
                verbose=False,
            )
            events = []

            main.run_experiment(
                args,
                progress_callback=lambda event, **values: events.append(
                    (event, values)
                ),
                compact_progress=True,
            )

        event_names = [event for event, _ in events]
        self.assertEqual(event_names.count("run_started"), 1)
        self.assertEqual(event_names.count("run_completed"), 1)
        self.assertIn("splitting", {
            values.get("stage") for event, values in events
            if event == "stage"
        })
        self.assertIn("evaluating", {
            values.get("stage") for event, values in events
            if event == "stage"
        })

    @patch(
        "main.sample_training_negative_edges",
        wraps=link_prediction.sample_training_negative_edges,
    )
    @patch("main.load_dataset")
    def test_asymmetric_run_expands_requested_regularizers(
        self, mock_load, mock_negative_sampling
    ):
        mock_load.return_value = self._graph()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "regularization.csv"
            args = SimpleNamespace(
                dataset="toy",
                out_filepath=str(output),
                data_root="unused",
                model_type=["asymmetric"],
                conv_layer_type="GCNConv",
                hidden_channels=4,
                out_channels=2,
                agae_regularization=["none", "latent-l2"],
                regularization_strength=0.1,
                learning_rate=0.005,
                weight_decay=0.01,
                query_size=1,
                train_ratio=0.8,
                epochs=1,
                seed=42,
                device="cpu",
                normalize_features=False,
            )

            main.run_experiment(args)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["regularization_type"] for row in rows},
            {"none", "latent-l2"},
        )
        self.assertEqual(
            {float(row["regularization_strength"]) for row in rows}, {0.1}
        )
        self.assertEqual({float(row["weight_decay"]) for row in rows}, {0.01})
        self.assertEqual(mock_negative_sampling.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in mock_negative_sampling.call_args_list],
            [args.seed, args.seed],
        )

if __name__ == "__main__":
    unittest.main()
