from .base import DataBase
import torch
import numpy as np
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


class GraphCFTData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(GraphCFTData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.dataset = getattr(self.cfg, 'dataset', 'MUTAG')

    def data_cls(self):
        data_cls = {
            'MUTAG': datasets.TUDataset(root=f'./data/MUTAG', name='MUTAG'),
            'PROTEINS':datasets.TUDataset(root=f'./data/PROTEINS', name='PROTEINS'),
        }
        return data_cls[self.dataset]

    def train_val_test_split(self, dataname, mode):
        split_plan = {
            'MUTAG': [0.4, 0.3],
            'PROTEINS': [0.3, 0.2]
        }
        torch.manual_seed(self.cfg.seed)
        dataset = self.data_cls()
        dataset = dataset.shuffle()
        train_num = int(len(dataset)*split_plan[dataname][0])
        val_num = int(len(dataset)*split_plan[dataname][1])
        train_dataset = dataset[:train_num]
        val_dataset = dataset[train_num:train_num + val_num]
        test_dataset = dataset[train_num+val_num:]

        dataset_loader = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        return dataset_loader[mode]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size,shuffle=False)

    @property
    def train_dataset(self):
        return self.train_val_test_split(self.dataset, 'train')

    @property
    def val_dataset(self):
        return self.train_val_test_split(self.dataset, 'val')

    @property
    def test_dataset(self):
        return self.train_val_test_split(self.dataset, 'val')


class NodeCFTData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(NodeCFTData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.dataset = getattr(self.cfg, 'dataset', 'Cora')

    def data_cls(self):
        data_cls = {
            'PubMed': datasets.Planetoid(root=f'./data/PubMed', name='PubMed'),
            'DBLP': datasets.CitationFull(root=f'./data/DBLP', name='DBLP'),
            'CiteSeer': datasets.Planetoid(root=f'./data/CiteSeer', name='CiteSeer'),
            'Cora': datasets.Planetoid(root=f'./data/Cora', name='Cora'),
            'CiteSeer1': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/CiteSeer1.pt'), None],
            'CiteSeer2': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/CiteSeer2.pt'), None],
            'CiteSeer3': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/CiteSeer3.pt'), None],
            'Cora1': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/Cora1.pt'), None],
            'Cora2': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/Cora2.pt'), None],
            'Cora3': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/Cora3.pt'), None]
        }
        data = data_cls[self.dataset][0]
        try:
            train_mask = data.train_mask
        except:
            train_mask, val_mask = self.train_val_split(data.y)
            data.train_mask = train_mask
            data.val_mask = val_mask
        return data

    def train_val_split(self, y):
        ids = np.arange(y.shape[0])
        y = y.numpy()
        x_train, x_test, _, _ = train_test_split(ids, y, test_size=0.3)
        train_mask, val_mask = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        train_mask[x_train] = 1
        val_mask[x_test] = 1
        train_mask = torch.tensor(train_mask).bool()
        val_mask = torch.tensor(val_mask).bool()
        return train_mask, val_mask

    def train_dataloader(self):
        return self.train_dataset.train_mask

    def val_dataloader(self):
        return self.train_dataset.val_mask

    def test_dataloader(self):
        # zheli mei cuo !!!!!!!!
        return self.train_dataset.val_mask

    @property
    def get_graph(self):
        return self.data_cls()

    @property
    def train_dataset(self):
        return self.data_cls()

    @property
    def val_dataset(self):
        return self.data_cls()

    @property
    def test_dataset(self):
        return self.data_cls()
