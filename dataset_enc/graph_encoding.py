import numpy as np
import torch
import torch.nn as nn
from torch_geometric import datasets
from torch_geometric.utils import add_self_loops
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
        'PubMed': datasets.Planetoid(root=f'.././data/PubMed', name='PubMed'),
        'DBLP': datasets.CitationFull(root=f'.././data/DBLP', name='DBLP'),
        'CiteSeer': datasets.Planetoid(root=f'.././data/CiteSeer', name='CiteSeer'),
        'Cora': datasets.Planetoid(root=f'.././data/Cora', name='Cora'),
        'CiteSeer1': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/CiteSeer1.pt'), None],
        'CiteSeer2': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/CiteSeer2.pt'), None],
        'CiteSeer3': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/CiteSeer3.pt'), None],
        'Cora1': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/Cora1.pt'), None],
        'Cora2': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/Cora2.pt'), None],
        'Cora3': [torch.load(f'/home/zhaohe/NNDF/graph_utils/data/Cora3.pt'), None]
    }
    return data_cls[name][0]


def svd_encoding(graph, device, hop=2, sample_num=100):
    x = graph.x.to(device)
    y = graph.y[graph.train_mask].to(device)
    edge_index = graph.edge_index.to(device)
    edge_index = add_self_loops(edge_index)[0]
    weights = torch.ones(edge_index.shape[1]).to(device).float()
    adj = torch.sparse_coo_tensor(edge_index, weights, size=(x.shape[0], x.shape[0])).to_dense()
    d = torch.sum(adj, dim=1).view(-1)
    d = torch.diag(d.pow(-0.5))
    A_hat = d @ adj @ d
    A_hat = A_hat + A_hat @ A_hat + A_hat @ A_hat @ A_hat
    U, S, V = torch.svd_lowrank(A_hat, q=1000)
    S = torch.diag(S)
    U = torch.matmul(U, torch.sqrt(S))
    V = torch.matmul(V, torch.sqrt(S))
    structure = torch.cat((U, V), dim=1)
    class_num = int(max(graph.y))+1
    structure_list = []
    feature_list = []
    for i in range(class_num):
        idx = torch.where(y==i)[0]
        structure_list.append(structure[idx])
        feature_list.append(x[idx])
    return structure_list, structure.shape[1], feature_list, x.shape[1]


def strucuture_svd_encoding(graph, device, latent_dim, niter):
    x = graph.x.to(device)
    y = graph.y[graph.train_mask].to(device)
    edge_index = graph.edge_index.to(device)
    edge_index = add_self_loops(edge_index)[0]
    weights = torch.ones(edge_index.shape[1]).to(device).float()
    adj = torch.sparse_coo_tensor(edge_index, weights, size=(x.shape[0], x.shape[0])).to_dense()
    d = torch.sum(adj, dim=1).view(-1)
    d = torch.diag(torch.sqrt(d))
    A_hat = d @ adj @ d
    A_hat = A_hat + A_hat @ A_hat + A_hat @ A_hat @ A_hat
    U, S, V = torch.svd_lowrank(A_hat, latent_dim, niter)
    S = torch.diag(S)
    U = torch.matmul(U, torch.sqrt(S))
    V = torch.matmul(V.T, torch.sqrt(S))
    features = torch.cat((U, V), dim=1)
    class_num = int(max(graph.y))+1
    structure_list = []

    for i in range(class_num):
        fi = features[torch.where(y==i)]
        structure_list.append(fi)
    return structure_list, x.shape[1]*2


def identity(graph, device):
    x = graph.x.to(device)
    y = graph.y[graph.train_mask].to(device)
    class_num = int(max(graph.y)) + 1
    feature_list = []
    for i in range(class_num):
        idx = torch.where(y == i)[0]
        feature_list.append(x[idx])
    return feature_list, x.shape[1], None, None

def knn_transform(graph, device='cuda'):
    x = graph.x.to(device)
    y = graph.y.to(device)
    edge_index = graph.edge_index.to(device)
    edge_index = add_self_loops(edge_index)[0]
    train_mask = graph.train_mask.to(device)
    labels = y[train_mask].to(device)
    class_num = int(torch.max(labels)+1)

    weights = torch.ones(edge_index.shape[1]).to(device)
    adj = torch.sparse_coo_tensor(edge_index, weights, size=(x.shape[0], x.shape[0])).to_dense()
    degree = torch.sum(adj, dim=1).view(-1,1)
    fx = torch.matmul(adj, x)/degree
    fx = fx[train_mask]

    encoding = []
    for i in range(class_num):
        idx = torch.where(labels==i)
        enc = fx[idx]
        encoding.append(enc)
    # encoding = torch.cat(encoding, dim=0)
    return encoding, x.shape[1]


def linear_embedding(graph, hidden_size=64, epochs=100):
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
    embs = embs.detach().cpu()
    features = []
    feat_list = []
    for i in range(class_num):
        features.append(embs[torch.where(labels==i)])
    return features, embs.shape[1], None, None

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

def graphmae(graph, name, device='cuda'):
    path = f'../param_data/{name}/GraphMAE_latent.pt'
    encoding = torch.load(path, map_location=device).detach()
    y = graph.y[graph.train_mask].to(device)
    class_num = int(max(graph.y)) + 1
    structure_list = []
    for i in range(class_num):
        idx = torch.where(y == i)[0]
        structure_list.append(encoding[idx])
    return structure_list, encoding.shape[1], None, None

def get_graph_encoding(name, device='cuda'):
    np.random.seed(1)
    # path = f'./param_data/{name}/deep_encoding.pt'
    data = load(name)
    # structure_enc, in_size, features, feature_size = svd_encoding(data, device)
    structure_enc, in_size, features, feature_size = identity(data, device)
    # structure_enc, in_size, features, feature_size = linear_embedding(data,hidden_size=64, epochs=200)
    # structure_enc, in_size, features, feature_size = graphmae(data, name, device)
    # F_feature, in_size = linear_embedding(data,hidden_size=64, epochs=200)
    # F_feature = knn_transform(data)
    # torch.save(F_feature, path)
    print(f'{name} Done')
    return structure_enc, in_size, features, feature_size
    # return F_feature, in_size
# main('Cora')
# # main('PubMed')
# # # main('DBLP')
# main('CiteSeer1')
# main('CiteSeer2')
# main('CiteSeer3')
#