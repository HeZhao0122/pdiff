import torch.nn as nn
from torchvision.datasets.vision import VisionDataset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import os
import torch
import math
import pdb
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from .base import DataBase
from torch.utils.data import Dataset, DataLoader
from core.utils import *
import warnings
import os
from torch_geometric import datasets

class PData(DataBase):
    def __init__(self, cfg, tasks, **kwargs):
        super(PData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.k = getattr(self.cfg, 'k', 200)
        self.batch_size = self.k
        self.token_length = getattr(self.cfg, 'token_length', 16)
        self.tf = self.cfg.tf
        self.task = tasks
        self.test_shapes = self.get_test_shape()

        # check the root path is  exist or not
        assert os.path.exists(self.root), f'{self.root} not exists'

        # check the root is a directory or file
        if os.path.isfile(self.root):
            state = torch.load(self.root, map_location='cpu')
            self.fix_model = state['model']
            self.fix_model.eval()
            self.fix_model.to('cpu')
            self.fix_model.requires_grad_(False)

            self.pdata = state['pdata']
            self.initials = state['initial']
            self.hidden_dims = state['hidden_dim'].float()
            self.mask = state['mask']
            self.label = state['label']
            self.accuracy = state['performance']
            self.train_layer = state['train_layer']
            self.param_dim = state['param_list']
            self.train_list = state['dataset_list']
            self.train_enc = self.construct_graph_enc(self.train_list)
            self.test_data = self.generate_test_loader(self.cfg.test_list, self.test_shapes, self.cfg.test_dim)

            # self.enc = self.get_graph_enc(self.datasets)
            # self.transform_latent()

        elif os.path.isdir(self.root):
            pass

    def get_test_shape(self):
        test_shapes = []
        for idx, task in self.task.items():
            test_shapes.append(task.get_param_shape())
        return test_shapes

    def generate_test_loader(self, datasets, test_shape, test_dim, num=20):
        '''
        for each dataset: num*len(test_dim)
        '''
        initials = []
        labels = []
        masks = []
        hidden_dim = []
        for i in range(len(datasets)):
            graph = load(datasets[i])
            feature_size = graph.x.shape[1]
            class_num = max(graph.y) + 1
            label = torch.tensor([i]*num*len(test_dim))
            labels.append(label)
            for j in range(len(test_dim)):
                mask = torch.cat([torch.ones([label.shape[0], test_shape[j]]), torch.zeros((label.shape[0], self.pdata.shape[1]- test_shape[j]))], dim=-1)
                masks.append(mask)
                # initial = torch.cat([torch.randn([label.shape[0], test_shape[j]]), torch.zeros((label.shape[0], self.pdata.shape[1]- test_shape[j]))], dim=-1)
                # initials.append(initial)
                for seed in range(label.shape[0]):
                    with TemporaryRandomSeed(seed):
                        ini = self.task[i].get_initialization()
                        padding = torch.zeros((1, self.pdata.shape[1] - ini.shape[1]))
                        initial = torch.cat([ini, padding], dim=-1)
                        initials.append(initial)
            for dim in test_dim:
                hidden_dim += [[dim, feature_size, class_num] for _ in range(num)]

        labels = torch.cat(labels)
        masks = torch.cat(masks, dim=0)
        initials = torch.cat(initials, dim=0)
        hidden_dim = torch.tensor(hidden_dim).float()
        enc = self.get_graph_enc(datasets)
        # test_pdata = torch.randn((labels.shape[0], self.pdata.shape[1]))
        encs = self.transform_latent(enc, labels)
        return initials, labels, encs, masks, hidden_dim, initials

    def construct_graph_enc(self, dataset_list):
        enc = self.get_graph_enc(dataset_list)
        return self.transform_latent(enc, self.label)

    def get_graph_enc(self, dataset_list):
        encoding = {}
        for idx, dataset in enumerate(dataset_list):
            path = f'./param_data/{dataset}/contrastive_encoding_identity.pt'
            print(f'load graph encoding from {path}')
            try:
                enc = torch.load(path, map_location='cpu')
                encoding[idx] = enc
            except:
                encoding[idx] = torch.zeros((1,100))
        return encoding

    def transform_latent(self, input_enc, label):
        enc_length = [enc.shape[1] for _, enc in input_enc.items()]
        pad_length = [max(enc_length)-l for l in enc_length]
        new_encs = {}
        # shape_list = {}
        for idx, enc in input_enc.items():
            m_enc = torch.mean(enc, dim=0)
            pad = torch.zeros(pad_length[idx])
            new_encs[idx] = torch.cat((m_enc,pad)).view(1, -1)
            # shape_tensor = torch.cat([torch.ones(enc.shape[0]), pad]).view(1, -1)

        encs = []
        for i in range(label.shape[0]):
            encs.append(new_encs[int(label[i])])
        encs = torch.cat(encs, dim=0)
        return encs

    def get_train_layer(self):
        return self.train_layer

    def get_model(self):
        return self.fix_model

    def get_accuracy(self):
        return self.accuracy

    def get_param_dim(self):
        return self.param_dim

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.pdata.shape[0], num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.pdata.shape[0], num_workers=self.num_workers, shuffle=False,
                          persistent_workers=True)

    @property
    def train_dataset(self):
        # return DataLoader(Parameters(self.pdata, self.k, split='train'), batch_size=self.batch_size, shuffle=True)
        return Parameters(self.pdata, self.label, self.train_enc, self.mask, self.hidden_dims, self.token_length, self.initials, split='train', tf=self.tf)

    @property
    def val_dataset(self):
        # return DataLoader(Parameters(self.pdata, self.k, split='val'), batch_size=self.batch_size, shuffle=True)
        return Parameters(self.pdata, self.label, self.train_enc, self.mask, self.hidden_dims, self.token_length,self.initials, split='val', tf=self.tf)

    @property
    def test_dataset(self):
        # return DataLoader(Parameters(self.pdata, self.k, split='test'), batch_size=self.batch_size, shuffle=True)
        return Parameters(self.test_data[0],self.test_data[1], self.test_data[2], self.test_data[3], self.test_data[4], self.token_length,self.test_data[5], split='test', tf=self.tf)


class Parameters(VisionDataset):
    # ToDo padding encoding data
    def __init__(self, batch, label, encoding=None, mask=None, hidden_dim=None, token_length=128, initial=None, split='train', tf=True):
        super(Parameters, self).__init__(root=None, transform=None, target_transform=None)
        if split == 'train':
            self.data = batch
        else:
            self.data = batch
        # data is a tensor list which is the parameters of the model
        self.label = label
        self.mask = mask
        self.hidden_dim = hidden_dim
        self.token_length = token_length
        self.initial = initial
        if tf:
            self.max_seq_len = math.ceil(self.data.shape[1]/token_length) * token_length
            self.padding()
        self.enc = encoding

    def padding(self):
        pad = torch.zeros(self.data.shape[0], self.max_seq_len-self.data.shape[1])
        data = torch.cat((self.data, pad), dim=1)
        ini = torch.cat((self.initial, pad), dim=1)
        if self.mask is not None:
            mask = torch.cat((self.mask, pad), dim=1)
            self.mask = mask.bool()
        data = data.view(self.data.shape[0], -1, self.token_length)
        ini = ini.view(self.data.shape[0], -1, self.token_length)
        self.data = data
        self.initial = ini


    def __getitem__(self, item):
        return self.data[item], self.mask[item], self.enc[item], self.label[item], self.hidden_dim[item], self.initial[item]

    def __len__(self) -> int:
        return len(self.data)


def load(name):
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
    return data_cls[name][0]



