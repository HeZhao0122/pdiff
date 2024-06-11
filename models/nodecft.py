
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class BaseGraph(torch.nn.Module):
    def __init__(self, in_feat, num_class, hidden_dim, aggr='gcn', layers=2):
        super(BaseGraph, self).__init__()
        layer_type = {
            'gcn':GCNConv,
            'gat':GATConv,
            'graphsage':SAGEConv
        }
        self.conv_type = layer_type[aggr]
        self.conv_layers = nn.Sequential()
        for i in range(layers-1):
            if i == 0:
                self.conv_layers.append(self.conv_type(in_feat, hidden_dim))
                # self.conv_layers.append(nn.ReLU())
            else:
                self.conv_layers.append(self.conv_type(hidden_dim, hidden_dim))
                # self.conv_layers.append(nn.ReLU())
        self.out = self.conv_type(hidden_dim, num_class)

    def forward(self, x, edge_index):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.out(x, edge_index)
        return x


class LinearGraph(torch.nn.Module):
    def __init__(self, in_feat, num_class, hidden_dim, aggr='gcn', layers=2):
        super(LinearGraph, self).__init__()
        layer_type = {
            'gcn':GCNConv,
            'gat':GATConv,
            'graphsage':SAGEConv
        }
        self.conv_type = layer_type[aggr]
        self.enc = nn.Linear(in_feat, hidden_dim)
        self.conv_layers = nn.Sequential()
        for i in range(layers):
            self.conv_layers.append(self.conv_type(hidden_dim, hidden_dim))
            # self.conv_layers.append(nn.ReLU())
        self.out = nn.Linear(hidden_dim, num_class)

    def forward(self, x, edge_index):
        x = self.enc(x)
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.out(x)
        return x


class EmbeddingGraph(torch.nn.Module):
    def __init__(self, node_num, num_class, hidden_dim, aggr='gcn', layers=2):
        super(EmbeddingGraph, self).__init__()
        layer_type = {
            'gcn':GCNConv,
            'gat':GATConv,
            'graphsage':SAGEConv
        }
        self.conv_type = layer_type[aggr]
        # self.enc = nn.Linear(in_feat, hidden_dim)
        self.embedding = nn.Parameter(torch.randn((node_num, hidden_dim)))
        self.conv_layers = nn.Sequential()
        for i in range(layers):
            self.conv_layers.append(self.conv_type(hidden_dim, hidden_dim))
            # self.conv_layers.append(nn.ReLU())
        self.out = nn.Linear(hidden_dim, num_class)

    def forward(self, x, edge_index):
        x = self.embedding
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.out(x)
        return x