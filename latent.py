import torch
import os
import seaborn as sns
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from tqdm import tqdm
path = './param_data/latent_no_shape.pt'
label_p = './param_data/dims_no_shape.pt'
latent = torch.load(path, map_location='cpu')
dims = torch.load(label_p, map_location='cpu')
latent = latent.reshape(latent.shape[0], -1).numpy()
dims = dims.numpy()

# dim_dict = {}
# for i in range(len(dims)):
#     dim = dims[i]
#     if dim not in dim_dict:
#         dim_dict[dim] = []
#     dim_dict[dim].append(latent[i])
#

def plot_embeddings(params, labels):
    color_list = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
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

plot_embeddings(latent, dims)


