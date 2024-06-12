import numpy as np
import torch
import torch.nn as nn
from torch_geometric import datasets
from torch_geometric.transforms import svd_feature_reduction


class LinearEmbedding(nn.Module):
    def __init__(self, feat_dim, hidden_dim, class_num):
        super(LinearEmbedding, self).__init__()
        self.emb = nn.Linear(feat_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        embedding = self.emb(x)
        # embedding = self.activation(embedding)
        out = self.classifier(embedding)
        return out

    def get_embedding(self, x):
        return self.emb(x)


def load(name):
    data_cls = {
        'PubMed': datasets.Planetoid(root=f'./data/PubMed', name='PubMed'),
        'DBLP': datasets.CitationFull(root=f'./data/DBLP', name='DBLP'),
        'CiteSeer': datasets.Planetoid(root=f'./data/CiteSeer', name='CiteSeer'),
        'Cora': datasets.Planetoid(root=f'./data/Cora', name='Cora')
    }
    return data_cls[name][0]


def fourier_transform(graph):
    x = graph.x
    edge_index = graph.edge_index
    weights = torch.ones(edge_index.shape[1])
    adj = torch.sparse_coo_tensor(edge_index, weights, size=(x.shape[0], x.shape[0])).to_dense()
    e,v = torch.linalg.eig(adj)
    fx = torch.matmul(v.float(), x)
    return fx


def knn_transform(graph):
    x = graph.x
    edge_index = graph.edge_index
    train_mask = graph.train_mask
    labels = graph.y[train_mask]
    class_num = int(torch.max(labels)+1)

    weights = torch.ones(edge_index.shape[1])
    adj = torch.sparse_coo_tensor(edge_index, weights, size=(x.shape[0], x.shape[0])).to_dense()
    self_loop = torch.eye(x.shape[0])
    adj = adj + self_loop
    degree = torch.sum(adj, dim=1).view(-1,1)
    fx = torch.matmul(adj, x)/degree
    fx = fx[train_mask]

    encoding = []
    for i in range(class_num):
        idx = torch.where(labels==i)
        enc = torch.mean(fx[idx], dim=0).view(1,-1)
        encoding.append(enc)
    encoding = torch.cat(encoding, dim=0)
    return encoding


def linear_embedding(graph, hidden_size=8*32, epochs=100):
    x = graph.x
    label = graph.y
    edge_index = graph.edge_index
    train_mask = graph.train_mask
    labels = graph.y[train_mask]
    class_num = int(torch.max(labels)+1)

    weights = torch.ones(edge_index.shape[1])
    adj = torch.sparse_coo_tensor(edge_index, weights, size=(x.shape[0], x.shape[0])).to_dense()
    self_loop = torch.eye(x.shape[0])
    adj = adj + self_loop
    degree = torch.sum(adj, dim=1).view(-1,1)
    fx = torch.matmul(adj, x)/degree

    classifier = LinearEmbedding(fx.shape[1], hidden_size, class_num)
    optim = torch.optim.SGD(classifier.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        optim.zero_grad()
        out = classifier(fx)
        loss = criterion(out[train_mask], label[train_mask])
        loss.backward()
        optim.step()
        print(f'Train loss: {loss}')
        if epoch % 5 == 0:
            pred = test(classifier, fx, label, graph.val_mask)
            print(f'Val Acc :{pred}')
            if pred > best_valid_acc:
                embs = classifier.get_embedding(fx)

    return embs.detach()

def test(net, x, label, mask):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
        outputs = net(x)
        predicted = outputs.argmax(dim=1)
        total += label[mask].size(0)
        correct += int((predicted[mask] == label[mask]).sum())
        print('Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        return 100. * correct / total


def main(name):
    np.random.seed(1)
    path = f'./param_data/{name}/deep_encoding.pt'
    data = load(name)
    F_feature = linear_embedding(data, hidden_size=512)
    torch.save(F_feature, path)
    print(f'{name} Done')

main('Cora')
main('PubMed')
# main('DBLP')
main('CiteSeer')