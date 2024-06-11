import torch
import os
import numpy as np

path = './param_data/latent.pt'
label_p = './param_data/labels.pt'
latent = torch.load(path, map_location='cpu')
labels = torch.load(label_p, map_location='cpu')
latent = latent.reshape(latent.shape[0], 1, -1)
d = {0:[], 1:[]}
for idx, l in enumerate(labels):
    d[int(l)].append(latent[idx])
print(latent)