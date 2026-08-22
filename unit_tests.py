import unittest
import torch
import utils
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
        ])
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
        data = dataset[0]
        #negative input should throw ValueError
        with self.assertRaises(ValueError):
            utils.inductive_split(data, query = -3)
        #float is invalid type
        with self.assertRaises(TypeError):
            utils.inductive_split(data, query = 1.3)

    def test_query_tensor(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
        ])
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
        data = dataset[0]
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
        ])
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
        data = dataset[0]
        #
        with self.assertRaises(TypeError):
            utils.inductive_split(data, [[2,3],[3,4]])
        utils.inductive_split(data, [633])

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

    def test_constructor(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
        ])
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
        data = dataset[0]
        link_prediction.LinkPrediction(data, conv_layer_type= "GCNConv", model_type = "asymmetric")


if __name__ == "__main__":
    unittest.main()
