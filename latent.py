import torch
import os
import seaborn as sns
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

datasets = ['CiteSeer', 'Cora']
path = './param_data/latent_cite_cora.pt'
label_p = './param_data/labels_cite_cora.pt'
latent = torch.load(path, map_location='cpu')
dims = torch.load(label_p, map_location='cpu')
latent = latent.reshape(latent.shape[0], -1)
for idx in range(len(datasets)):
    encoding_path = f'./param_data/{datasets[idx]}/contrastive_encoding_identity.pt'
    encoding = torch.load(encoding_path, map_location='cpu').view(1,-1)
    encoding = encoding.expand(50, encoding.shape[1])
    latent = torch.cat([latent, encoding], dim=0)
    dims = torch.cat([dims, torch.tensor([idx + 2]*50)], dim=0)
# dim_dict = {}
# for i in range(len(dims)):
#     dim = dims[i]
#     if dim not in dim_dict:
#         dim_dict[dim] = []
#     dim_dict[dim].append(latent[i])
latent = torch.nn.functional.normalize(latent, p=2, dim=1)
latent = latent.numpy()
dims = dims.numpy()

def plot_embeddings(params, labels):
    color_list = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    x_min, x_max = np.min(params, 0), np.max(params, 0)
    data = (params-x_min)/(x_max-x_min)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(data)
    label_set = np.unique(labels)
    # fig = plt.figure()
    # ax = plt.axes()
    for i in range(len(label_set)):
        plt.scatter(Y[labels == label_set[i], 0], Y[labels == label_set[i], 1], color=color_list[i])
    # ax.scatter(Y[:, 0], Y[:, 1])
    plt.show()

def PCA_visual(params, labels):
    color_list = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    x_min, x_max = np.min(params, 0), np.max(params, 0)
    data = (params - x_min) / (x_max - x_min)
    pca = PCA(n_components=2)
    Y = pca.fit_transform(data)
    label_set = np.unique(labels)
    # fig = plt.figure()
    # ax = plt.axes()
    for i in range(len(label_set)):
        plt.scatter(Y[labels == label_set[i], 0], Y[labels == label_set[i], 1], color=color_list[i])
    # ax.scatter(Y[:, 0], Y[:, 1])
    plt.show()

# plot_embeddings(latent, dims)
PCA_visual(latent, dims)


