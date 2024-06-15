import os

import numpy as np
import torch
import statistics
from tqdm import tqdm
from torch_geometric import datasets
device = 'cpu'

def load(name):
    data_cls = {
        'PubMed': datasets.Planetoid(root=f'./data/PubMed', name='PubMed'),
        'DBLP': datasets.CitationFull(root=f'./data/DBLP', name='DBLP'),
        'CiteSeer': datasets.Planetoid(root=f'./data/CiteSeer', name='CiteSeer'),
        'Cora': datasets.Planetoid(root=f'./data/Cora', name='Cora')
    }
    return data_cls[name][0]


def load_params(dataset, lr, m, seed, hidden_dim):
    path = f'./param_data/{dataset}/all_seed_all/{seed}/{hidden_dim}_{lr}_{m}/data.pt'
    state = torch.load(path, map_location=torch.device(device))
    params = state['pdata']
    performance = state['performance']
    best = max(performance)
    median = statistics.median(performance)
    mean = statistics.mean(performance)

    return params, [best, mean, median]


def merge_dataset(dataset, seed_num):
    # use seed 0 to replace org data.pt
    path = f'./param_data/{dataset}/all_seed_all/0/data.pt'
    state = torch.load(path)
    all_params = []
    best_all, median_all, mean_all = [],[],[]
    for seed in tqdm(range(seed_num)):
        params, (best, mean, median) = load_params(dataset, seed)
        all_params.append(params)
        best_all.append(best)
        median_all.append(median)
        mean_all.append(mean)
    all_params = torch.cat(all_params, dim=0).numpy()
    # np.random.shuffle(all_params)
    all_params = torch.from_numpy(all_params)
    state['pdata'] = all_params
    new_path = f'./param_data/{dataset}/data.pt'
    torch.save(state, new_path)
    print(statistics.mean(best_all))
    print(statistics.mean(median_all))
    print(statistics.mean(mean_all))

def load_state_dict(dataset_list):
    state_dict = {}
    for idx, dataset in enumerate(dataset_list):
        path = f'./param_data/{dataset}/all_seed_all/0/64_0.1_0.9/data.pt'
        state = torch.load(path, map_location=torch.device(device))
        state_dict['train_layer'] = state['train_layer']
        state_dict[dataset] = state['model']
        state_dict['model'] = state['model']
        state_dict['param_list'] = state['param_list']
    return state_dict


def merge_all(dataset_list, lrs, moms, seeds, hidden_dims):
    all_param = []
    all_label = []
    length = []
    dims = []
    for idx, dataset in enumerate(dataset_list):
        graph = load(dataset)
        feature_size = graph.x.shape[1]
        class_num = max(graph.y) + 1
        data_param = []
        data_performance = []
        dim_sets = []
        for seed in seeds:
            for lr in lrs:
                for m in moms:
                    for dim in hidden_dims:
                        param, performance = load_params(dataset, lr, m, seed, dim)
                        data_param.append(param)
                        data_performance.append(performance[1])
                        dim_sets += [dim] * param.shape[0]
        # data_param = torch.cat(data_param, dim=0)
        dims += [[d, feature_size, class_num] for d in dim_sets]
        label = torch.tensor([idx]*sum([p.shape[0] for p in data_param]))
        print(f'{dataset}: {statistics.mean(data_performance)}')
        all_param += data_param
        all_label.append(label)
        length += [p.shape[1] for p in data_param]
    padding_length = [max(length)-i for i in length]
    print(max(length))

    padded_param = []
    padded_mask = []

    for pad_length, param in zip(padding_length, all_param):
        mask = torch.ones(param.shape)
        padding = torch.zeros((param.shape[0], pad_length))
        param = torch.cat((param, padding), dim=1)
        padded_param.append(param)

        mask = torch.cat((mask, padding), dim=1)
        padded_mask.append(mask)

    padded_mask = torch.cat(padded_mask, dim=0)
    padded_param = torch.cat(padded_param, dim=0)
    all_label = torch.cat(all_label)
    dims = torch.tensor(dims)
    state_dict = load_state_dict(['PubMed'])
    state_dict['pdata'] = padded_param
    state_dict['mask'] = padded_mask
    state_dict['label'] = all_label
    state_dict['hidden_dim'] = dims
    state_dict['performance'] = []
    state_dict['dataset_list'] = dataset_list

    new_path = f'./param_data/PubMed/data.pt'
    torch.save(state_dict, new_path)



lrs = [0.1, 0.05]
momentum = [0.9, 0.95]
seeds = [0, 1, 2]
hidden_dims = [32, 50, 64]
dataset_list = ['PubMed']
merge_all(dataset_list, lrs, momentum, seeds, hidden_dims)
# merge_dataset(dataset, seed_num)
