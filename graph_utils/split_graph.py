import random
import torch
import numpy as np
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


def get_graph(name):
    data_cls = {
        'PubMed': datasets.Planetoid(root=f'./data/PubMed', name='PubMed'),
        'DBLP': datasets.CitationFull(root=f'./data/DBLP', name='DBLP'),
        'CiteSeer': datasets.Planetoid(root=f'./data/CiteSeer', name='CiteSeer'),
        'Cora': datasets.Planetoid(root=f'./data/Cora', name='Cora')
    }
    data = data_cls[name][0]
    return data


def extract_subgraph(labels, graph):
    nodes = []
    y = graph.y
    for label in labels:
        label = torch.tensor(label)
        nodes.append(torch.where(y==label)[0])
    nodes = torch.cat(nodes)
    subgraph = graph.subgraph(nodes)
    return subgraph

def label_mapping(org_label, y):
    mapping = {}
    idx = 0
    for label in org_label:
        mapping[label] = idx
        idx += 1

    for i in range(len(y)):
        y[i] = mapping[int(y[i])]

    return y



def splitG(graph, num=2):
    x, y = graph.x, graph.y
    class_num = int(max(y)) + 1
    labels = [i for i in range(class_num)]
    # label1 = random.choices(labels, k=int(0.5*class_num))
    label1 = [0,3]
    label2 = [1,4]
    label3 = set(labels) - set(label1)-set(label2)

    sub1 = extract_subgraph(label1, graph)
    sub2 = extract_subgraph(label2, graph)
    sub3 = extract_subgraph(label3, graph)

    sub1.y = label_mapping(label1, sub1.y)
    sub2.y = label_mapping(label2, sub2.y)
    sub3.y = label_mapping(label3, sub3.y)


    graphs = {'Cora1': sub1,
              'Cora2': sub2,
              'Cora3': sub3}

    return graphs


def get_sbugraphs(name, num=2):
    org_graph = get_graph(name)
    if num==0:
        return {name: org_graph}
    subgraphs = splitG(org_graph, name)

    return subgraphs