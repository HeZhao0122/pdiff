import seaborn as sns
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt


def min_max_norm(input):
    min = torch.min(input)
    max = torch.max(input)
    x = (input - min) / (max - min)
    return x


def load_param(dataset):
    path = f'./param_data/{dataset}/data.pt'
    file = torch.load(path)
    param = file['pdata'][0].reshape((32,-1))

    return param


dataset = 'cifar10'
k = 100
params = load_param(dataset)
n_param = pd.DataFrame(min_max_norm(params))
# n_param = pd.DataFrame(params)


plot = sns.heatmap(n_param)
plt.show()