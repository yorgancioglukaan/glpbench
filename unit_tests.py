import unittest
from unittest.mock import patch
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import utils
import datasets
import main
from experiment_logging import CSVResultLogger, ExperimentResult
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv
import link_prediction

class TestInductiveSplit(unittest.TestCase):
    @staticmethod
    def _undirected_data(num_nodes, edges):
        directed_edges = edges + [(target, source) for source, target in edges]
        return Data(
            edge_index=torch.tensor(directed_edges, dtype=torch.long).t().contiguous(),
            num_nodes=num_nodes,
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
        utils.inductive_split(data, [5])

    def test_inductive_negatives_do_not_overlap_positive_edges(self):
        data = self._undirected_data(
            7,
            [(0, 5), (1, 6), (0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        )
        query_nodes = {5, 6}

        _, _, test_transductive, test_inductive = utils.generate_comparable_splits(
            data, query=list(query_nodes), train_ratio=0.75
        )

        num_positive = int(test_inductive.edge_label.sum().item())
        negative_edges = test_inductive.edge_label_index[:, num_positive:]
        positive_edges = {
            tuple(sorted(edge)) for edge in data.edge_index.t().tolist()
        }

        self.assertTrue(torch.equal(test_inductive.edge_index, test_transductive.edge_index))
        self.assertEqual(
            test_inductive.edge_label_index.size(1),
            test_transductive.edge_label_index.size(1),
        )
        self.assertEqual(negative_edges.size(1), num_positive)
        self.assertEqual(test_transductive.edge_label_index.size(1), 2 * num_positive)
        for source, target in negative_edges.t().tolist():
            self.assertIn(target, query_nodes)
            self.assertNotIn(source, query_nodes)
            self.assertNotEqual(source, target)
            self.assertFalse(source in query_nodes and target in query_nodes)
            self.assertNotIn(tuple(sorted((source, target))), positive_edges)

    def test_inductive_negative_sampling_raises_when_no_candidate_exists(self):
        data = self._undirected_data(
            4, [(0, 3), (1, 3), (2, 3), (0, 1), (1, 2), (0, 2)]
        )

        with self.assertRaisesRegex(ValueError, "no eligible non-neighbor"):
            utils.inductive_split(data, query=[3])
        

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

    def test_model_construction_for_supported_convolutions(self):
        layer_types = {
            "GCNConv": GCNConv,
            "GraphConv": GraphConv,
            "SAGEConv": SAGEConv,
        }
        for model_class in (link_prediction.GAE_Model, link_prediction.AGAE_Model):
            for layer_name, layer_class in layer_types.items():
                with self.subTest(model=model_class.__name__, layer=layer_name):
                    model = model_class(3, 4, 2, conv_layer_type=layer_name)
                    self.assertIsInstance(model.conv1, layer_class)
                    self.assertIsInstance(model.conv2, layer_class)

    def test_invalid_model_and_convolution_types(self):
        data = self._training_data()
        with self.assertRaises(ValueError):
            link_prediction.GAE_Model(3, 4, 2, conv_layer_type="InvalidConv")
        with self.assertRaises(ValueError):
            link_prediction.AGAE_Model(3, 4, 2, conv_layer_type="InvalidConv")
        with self.assertRaises(ValueError):
            link_prediction.LinkPrediction(data, model_type="invalid")

    def test_encoder_output_shapes_are_finite(self):
        data = self._training_data()
        for model_class in (link_prediction.GAE_Model, link_prediction.AGAE_Model):
            for layer_name in ("GCNConv", "GraphConv", "SAGEConv"):
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
            self.assertEqual(json.loads(rows[0]["extra_fields"]), {"learning_rate": 0.01})

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
            {"train_ratio": 1.1},
            {"roc_auc": -0.1},
            {"loss": -1.0},
        ]
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self._result(**values)

    def test_logger_requires_experiment_result(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                CSVResultLogger(Path(directory) / "results.csv").log({})


class TestDatasetLoading(unittest.TestCase):
    @staticmethod
    def _graph():
        return Data(
            x=torch.tensor([[2.0, 2.0], [1.0, 3.0], [4.0, 0.0]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )

    def test_available_datasets_has_stable_names(self):
        self.assertEqual(datasets.available_datasets(), ["citeseer", "cora", "pubmed"])

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

    @patch("main.load_dataset")
    def test_unspecified_model_type_runs_both_models_on_both_splits(self, mock_load):
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
        })
        self.assertEqual({row["run_id"] for row in rows}, {run_id})

if __name__ == "__main__":
    unittest.main()
