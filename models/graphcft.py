import os
import torch.nn as nn

from torch.nn import Linear

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, GATConv


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
        for i in range(layers):
            if i == 0:
                self.conv_layers.append(self.conv_type(in_feat, hidden_dim))
                # self.conv_layers.append(nn.ReLU())
            else:
                self.conv_layers.append(self.conv_type(hidden_dim, hidden_dim))
                # self.conv_layers.append(nn.ReLU())
        self.out = nn.Linear(hidden_dim, num_class)

    def forward(self, x, edge_index, batch):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.out(x)
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
        self.conv_layers = nn.Sequential()
        self.enc = nn.Linear(in_feat, hidden_dim)
        for i in range(layers):
            self.conv_layers.append(self.conv_type(hidden_dim, hidden_dim))
            # self.conv_layers.append(nn.ReLU())
        self.out = nn.Linear(hidden_dim, num_class)

    def forward(self, x, edge_index, batch):
        x = self.enc(x)
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.out(x)
        return x