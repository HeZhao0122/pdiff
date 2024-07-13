import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from models.nodecft import BaseGraph
import datetime
from split_graph import get_sbugraphs
from core.utils import *
import json
from statistics import median, mean


def train_for_data(dataset, seed, lr, momentum,  hidden_dim, weight_decay,graph, epochs=200, save_num=25):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_mask = graph.train_mask
    val_mask = graph.val_mask
    class_num = int(max(graph.y)) + 1
    net = BaseGraph(in_feat=graph.x.shape[1], num_class=class_num, hidden_dim=hidden_dim, aggr='gat', layers=3)
    optimizer = SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    all_epoch = epochs + save_num

    best_acc = 0
    train_layer = [name for name, module in net.named_parameters()]
    shape_list = [pa.view(-1).shape[0] for name, pa in net.named_parameters() if name in train_layer]
    print(sum(shape_list))

    data_path = '.././param_data'

    tmp_path = os.path.join(data_path, dataset)
    tmp_path = os.path.join(tmp_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    final_path = os.path.join(data_path, dataset)
    final_path = os.path.join(final_path,
                              f'all_seed_all/{seed}/{hidden_dim}_{lr}_{momentum}')

    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(final_path, exist_ok=True)

    net = net.cuda()

    save_model_accs = []
    parameters = []
    initialization = state_part(train_layer, net)
    init_param = []
    for key in train_layer:
        init_param.append(initialization[key].reshape(-1))
    init_param = torch.cat(init_param, 0)

    for i in range(0, all_epoch):
        train(net, criterion, optimizer, train_mask, graph)
        acc = test(net, criterion, val_mask, graph)
        best_acc = max(acc, best_acc)
        if i == (epochs - 1):
            # print("saving the model")
            torch.save(net, os.path.join(tmp_path, "whole_model.pth"))
            fix_partial_model(train_layer, net)
            parameters = []
        if i >= epochs:
            parameters.append(state_part(train_layer, net))
            save_model_accs.append(acc)
            if len(parameters) == 5 or i == all_epoch - 1:
                torch.save(parameters, os.path.join(tmp_path, "p_data_{}.pt".format(i)))
                parameters = []

    print(best_acc)

    pdata = []
    param_list = get_param_dims(train_layer, net)
    for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
        buffers = torch.load(file)
        for buffer in buffers:
            param = []

            for key in train_layer:
                param.append(buffer[key].reshape(-1))
            param = torch.cat(param, 0)
            pdata.append(param)
    batch = torch.stack(pdata)
    mean = torch.mean(batch, dim=0)
    std = torch.std(batch, dim=0)
    print(f'param shape:{batch.shape}')

    state_dic = {
        'initialization': init_param.cpu().detach(),
        'pdata': batch.cpu().detach(),
        'mean': mean.cpu(),
        'std': std.cpu(),
        'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
        'train_layer': train_layer,
        'param_list': param_list,
        'performance': save_model_accs,
    }

    torch.save(state_dic, os.path.join(final_path, "data.pt"))
    json_state = {
        'performance': save_model_accs
    }
    json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

    # copy the code file(the file) in state_save_dir
    shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                        os.path.basename(__file__)))

    # delete the tmp_path
    shutil.rmtree(tmp_path)
    print(f'best model performance: {max(save_model_accs)}')
    print(f'mean model performance: {sum(save_model_accs) / len(save_model_accs)}')
    print(f'median model performance: {median(save_model_accs)}')
    print("data process over")



def train(net, criterion, optimizer, mask, graph):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    data = graph.cuda()
    # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
    optimizer.zero_grad()
    outputs = net(data.x, data.edge_index)
    loss = criterion(outputs[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    predicted = outputs.argmax(dim=1)
    total += data.y[mask].size(0)
    correct += int((predicted[mask] == data.y[mask]).sum())

    # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss, 100. * correct / total, correct, total))


def test(net, criterion, mask, graph):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        data = graph.cuda()
        # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
        outputs = net(data.x, data.edge_index)
        loss = criterion(outputs[mask], data.y[mask])

        test_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += data.y[mask].size(0)
        correct += int((predicted[mask] == data.y[mask]).sum())
        # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss, 100. * correct / total, correct, total))
        return 100. * correct / total


def main():
    subgraphs = get_sbugraphs('Cora', num=2)
    seeds = [0,1,2]
    lrs = [0.1, 0.05]
    weight_decay = 0.0005
    momentums = [0.95, 0.9]
    hidden_dim = 32
    for name, subgraph in subgraphs.items():
        torch.save(subgraph, f'./data/{name}.pt')
        for seed in seeds:
            for lr in lrs:
                for momentum in momentums:
                    train_for_data(name, seed, lr, momentum, hidden_dim, weight_decay, subgraph)

main()
# 119782 citeseer1 2
# 119922 citeseer
# 17321 PubMed 32
# 47317 Cora
# 47142 Cora 1 2
# 47177 cora 3
