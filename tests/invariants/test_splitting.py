import unittest
from unittest.mock import patch
import argparse
import copy
import csv
from contextlib import redirect_stdout
from dataclasses import replace
import importlib
from io import StringIO
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import torch
import yaml
import glpbench.protocols.splitting as utils
import glpbench.datasets.builtin as datasets
import glpbench.runners.experiment as main
import glpbench.runners.benchmark as benchmark
import glpbench.runners.audit as convergence_audit
import glpbench.runners.search as search
import glpbench.preprocessing.reduction as feature_reduction
import glpbench.training.scheduling as lr_scheduling
from glpbench import preprocessing, training
import glpbench.datasets as glp_datasets
import glpbench.evaluation as glp_evaluation
import glpbench.logging as glp_logging
import glpbench.models as glp_models
import glpbench.protocols as glp_protocols
import glpbench.runners.experiment as experiment_runner
import glpbench.runners.benchmark as benchmark_runner
import glpbench.runners.search as search_runner
import glpbench.runners.audit as audit_runner
import glpbench.evaluation.pipeline as experiment_pipeline
from glpbench.logging.csv_store import (
    CSVEpochHistoryLogger,
    CSVResultLogger,
    EpochHistoryRecord,
    ExperimentResult,
    make_configuration_id,
)
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, SAGEConv
import glpbench.models.link_prediction as link_prediction


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
        splits = utils.generate_comparable_splits(
            data, query_size=1, seed=11, train_ratio=0.75
        )
        train, val = splits.train, splits.validation
        test_transductive = splits.tests["transductive"]
        test_inductive = splits.tests["semi-inductive"]

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

    def test_comparable_splits_use_named_bundle_access(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 0), (0, 4), (1, 5), (2, 6), (3, 7)],
        )

        splits = utils.generate_comparable_splits(
            data, query=[7], validation_query=[6], seed=3, train_ratio=0.75
        )
        self.assertIsInstance(splits, utils.SplitBundle)
        self.assertEqual(
            set(splits.tests), {"transductive", "semi-inductive"}
        )
        self.assertEqual(splits.metadata.name, "matched-semi-inductive")

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
            (splits.tests["transductive"], "test", "transductive"),
            (splits.tests["semi-inductive"], "test", "semi-inductive"),
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

        first_partitions = (
            first.train,
            first.validation,
            first.tests["transductive"],
            first.tests["semi-inductive"],
        )
        second_partitions = (
            second.train,
            second.validation,
            second.tests["transductive"],
            second.tests["semi-inductive"],
        )
        for first_split, second_split in zip(first_partitions, second_partitions):
            for attribute in (
                "edge_index", "edge_label_index", "edge_label",
                "visible_node_mask", "validation_query_mask",
                "test_query_mask",
            ):
                self.assertTrue(torch.equal(
                    getattr(first_split, attribute),
                    getattr(second_split, attribute),
                ))

    def test_validation_only_splits_match_comparable_training_and_validation(self):
        data = self._undirected_data(
            30,
            [
                (source, target)
                for source in range(30)
                for target in range(source + 1, 30)
                if (source * 7 + target * 11) % 5 < 2
            ],
        )
        kwargs = dict(
            query_size=2,
            validation_query_size=2,
            train_ratio=0.8,
            seed=21,
        )

        comparable = utils.generate_comparable_splits(data, **kwargs)
        validation_only = utils.generate_validation_splits(data, **kwargs)

        self.assertIsInstance(validation_only, utils.SplitBundle)
        self.assertEqual(validation_only.tests, {})
        for expected, actual in (
            (comparable.train, validation_only.train),
            (comparable.validation, validation_only.validation),
        ):
            for attribute in (
                "edge_index", "edge_label_index", "edge_label",
                "visible_node_mask", "validation_query_mask",
                "test_query_mask",
            ):
                self.assertTrue(torch.equal(
                    getattr(expected, attribute), getattr(actual, attribute)
                ))

    def test_comparable_split_generation_runs_end_to_end_audit(self):
        data = self._undirected_data(
            8,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 6), (3, 7)],
        )
        with patch(
            "glpbench.protocols.splitting.audit_comparable_splits",
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
                data, replace(clean, validation=validation)
            )

        inductive = clean.tests["semi-inductive"].clone()
        negative_index = int((inductive.edge_label == 1).sum())
        inductive.edge_label_index[:, negative_index] = data.edge_index[:, 0]
        with self.assertRaisesRegex(ValueError, "negative edge that is positive"):
            utils.audit_comparable_splits(
                data,
                replace(
                    clean,
                    tests={**clean.tests, "semi-inductive": inductive},
                ),
            )

        transductive = clean.tests["transductive"].clone()
        transductive.edge_label_index = transductive.edge_label_index[:, :-1]
        transductive.edge_label = transductive.edge_label[:-1]
        with self.assertRaisesRegex(ValueError, "different labeled-edge counts"):
            utils.audit_comparable_splits(
                data,
                replace(
                    clean,
                    tests={**clean.tests, "transductive": transductive},
                ),
            )

        validation = clean.validation.clone()
        validation_positive = validation.edge_label == 1
        validation.edge_label_index[:, validation_positive] = (
            clean.tests["semi-inductive"].edge_label_index[
                :, clean.tests["semi-inductive"].edge_label == 1
            ][:, :int(validation_positive.sum())]
        )
        with self.assertRaisesRegex(ValueError, "Positive edges overlap"):
            utils.audit_comparable_splits(
                data, replace(clean, validation=validation)
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

        partitions = (
            splits.train,
            splits.validation,
            *splits.tests.values(),
        )
        for split in partitions:
            represented = (
                utils._canonical_edge_pairs(split.edge_index)
                | utils._canonical_edge_pairs(split.edge_label_index)
            )
            self.assertNotIn(query_pair, represented)
        positive_sets = [
            utils._canonical_edge_pairs(
                split.edge_label_index[:, split.edge_label == 1]
            )
            for split in partitions
        ]
        for index, left in enumerate(positive_sets):
            for right in positive_sets[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertTrue(utils.audit_comparable_splits(data, splits).passed)

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
        self.assertIsInstance(splits, utils.SplitBundle)
        train, val = splits.train, splits.validation
        test = splits.tests["semi-inductive"]
        self.assertIs(splits.train, train)
        self.assertIs(splits.validation, val)
        self.assertIs(splits.tests["semi-inductive"], test)
        expected_visible_mask = ~torch.tensor([
            False, False, False, False, True, True
        ])
        for split, stage, setting in (
            (train, "train", "transductive"),
            (val, "validation", "semi-inductive"),
            (test, "test", "semi-inductive"),
        ):
            self.assertEqual(split.split_stage, stage)
            self.assertEqual(split.split_setting, setting)
            self.assertTrue(torch.equal(
                split.visible_node_mask, expected_visible_mask
            ))

    def test_inductive_negatives_do_not_overlap_positive_edges(self):
        data = self._undirected_data(
            10,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 3), (1, 4), (2, 5), (0, 6), (2, 7), (1, 8), (3, 9)],
        )
        validation_query_nodes = {6, 7}
        test_query_nodes = {8, 9}
        query_nodes = validation_query_nodes | test_query_nodes

        splits = utils.generate_comparable_splits(
            data,
            query=list(test_query_nodes),
            validation_query=list(validation_query_nodes),
            train_ratio=0.75,
            seed=9,
        )
        train, val = splits.train, splits.validation
        test_transductive = splits.tests["transductive"]
        test_inductive = splits.tests["semi-inductive"]

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
        for split, stage, setting in (
            (train, "train", "transductive"),
            (val, "validation", "semi-inductive"),
            (test_transductive, "test", "transductive"),
            (test_inductive, "test", "semi-inductive"),
        ):
            self.assertEqual(split.split_stage, stage)
            self.assertEqual(split.split_setting, setting)
            self.assertTrue(torch.equal(
                split.visible_node_mask, ~expected_query_mask
            ))
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

