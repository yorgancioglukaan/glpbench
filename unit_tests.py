import unittest
import torch
import utils
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

class TestInductiveSplit(unittest.TestCase):
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
        

if __name__ == "__main__":
    unittest.main()