import torch
import torch.nn.functional as f
import seaborn as sns
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

def load(path, idx=0):
    state_dict = torch.load(path)
    params = state_dict['pdata']
    # normalize_param = f.normalize(params, dim=1, p=2)
    normalize_param = params
    return normalize_param[idx]

def draw(param1, param2):
    param1 = param1.view(1,-1)
    param2 = param2.view(1,-1)
    sns.set_theme()

    print(f.cosine_similarity(param1, param2))


def plot_embeddings(params, ):
    x_min, x_max = np.min(params, 0), np.max(params, 0)
    data = (params-x_min)/(x_max-x_min)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(data)
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(Y[:, 0], Y[:, 1])
    plt.show()

def plot_norms(params, ):
    # x_min, x_max = np.min(params, 0), np.max(params, 0)
    # data = (params-x_min)/(x_max-x_min)
    norm = params.norm(p=2,dim=1).numpy()
    fig = plt.figure()
    ax = plt.axes()
    ax.hist(norm)
    plt.show()


def main(dataset):
    seeds = 500
    params = []
    for seed in tqdm(range(seeds)):
        path = f'./param_data/{dataset}/all_seed/{seed}/data.pt'
        param = load(path, 0).view(1, -1)
        params.append(param)
    params = torch.cat(params, dim=0)
    # plot_embeddings(params)
    plot_norms(params)


dataset1 = 'PubMed'
dataset2 = 'CiteSeer'
seed1 = 0
seed2 = 4
path1 = f'./param_data/{dataset1}/all_seed/{seed1}/data.pt'
path2 = f'./param_data/{dataset1}/all_seed/{seed2}/data.pt'
main(dataset1)


