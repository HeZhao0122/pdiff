import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random


class CL(nn.Module):
    def __init__(self, in_dim, latent_dim, device='cuda'):
        super(CL, self).__init__()
        # self.param_encoder = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.ELU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )
        self.encoder = nn.Sequential(
            nn.Linear(in_dim,latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim*4),
            nn.Tanh(),
            nn.Linear(latent_dim*4, latent_dim),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, pos, neg):
        pos_latent, pos_target = pos
        neg_latent, neg_target = neg
        # pos_latent = self.param_encoder(pos_latent)
        # neg_latent = self.param_encoder(neg_latent)
        pos_target = self.encoder(pos_target)
        neg_target = self.encoder(neg_target)
        shape = pos_latent.shape

        pos_score = pos_latent * pos_target
        neg_score = neg_latent * neg_target

        loss = self.criterion(pos_score, torch.ones(shape).to(self.device))
        loss = loss + self.criterion(neg_score, torch.zeros(shape).to(self.device))
        return loss

    def get_encoding(self, encoding):
        return self.encoder(encoding)


class ContrastivePair(Dataset):
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def __getitem__(self, item):
        return [self.pos[0][item], self.pos[1][item]], [self.neg[0][item], self.neg[1][item]]

    def __len__(self):
        return self.pos[0].shape[0]


def load_latent(root, device='cpu'):
    latents = torch.load(os.path.join(root, 'latent_all.pt'), map_location=device)
    latents = latents.view(latents.shape[0], -1)
    labels = torch.load(os.path.join(root, 'labels_all.pt'), map_location=device)
    dataset_num = int(torch.max(labels)) + 1
    res = [[] for _ in range(dataset_num)]
    for idx, l in enumerate(labels):
        res[int(l)].append(latents[idx].view(1, -1))
    print('finish loading!')
    return res


def get_graph_enc(dataset_list, device='cpu'):
    encoding = {}
    for idx, dataset in enumerate(dataset_list):
        path = f'./param_data/{dataset}/knn_encoding.pt'
        enc = torch.load(path, map_location=device)
        encoding[idx] = torch.mean(enc, dim=0).view(1, -1)

    return transform_latent(encoding, device)


def transform_latent(input_enc, device):
    enc_length = [enc.shape[1] for _, enc in input_enc.items()]
    pad_length = [max(enc_length)-l for l in enc_length]
    new_encs = {}
    for idx, enc in input_enc.items():
        m_enc = torch.mean(enc,dim=0)
        pad = torch.zeros(pad_length[idx]).to(device)
        new_encs[idx] = torch.cat((m_enc,pad)).view(1, -1)
    return new_encs, max(enc_length)

def get_pos_neg_pair(graph_encoding, latents, num=100):
    pos_pair, neg_pair = [], []
    for idx, enc in graph_encoding.items():
        pos = random.choices(latents[idx], k=num)
        target = [graph_encoding[idx] for _ in range(num)]
        pos_pair += [pos, target]

        neg_set = [i for i in graph_encoding if i != idx]
        sample_num = int(num/len(neg_set))
        negs = []
        for i in neg_set:
            neg = random.choices(latents[i], k=sample_num)
            negs += neg
        neg_pair += [negs, target]
    pos_pair = [torch.cat(pos_pair[0], dim=0), torch.cat(pos_pair[1], dim=0)]
    neg_pair = [torch.cat(neg_pair[0], dim=0), torch.cat(neg_pair[1], dim=0)]
    return pos_pair, neg_pair


def train(dataloder, epochs, lr, in_dim, latent_dim, device):
    model = CL(in_dim, latent_dim, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for pos, neg in dataloder:
            pos = [pos[0].to(device), pos[1].to(device)]
            neg = [neg[0].to(device), neg[1].to(device)]
            optimizer.zero_grad()
            loss = model(pos, neg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch} loss: {epoch_loss}')
    print('Training Done!')
    return model


def main():
    device = 'cuda'
    latents = load_latent('./param_data/')
    encodings, in_size = get_graph_enc(['PubMed', 'CiteSeer'])
    pos_pair, neg_pair = get_pos_neg_pair(encodings, latents, num=75)
    train_data = ContrastivePair(pos_pair, neg_pair)
    train_loader = DataLoader(train_data, batch_size=50, num_workers=1, shuffle=True, persistent_workers=True)
    model = train(train_loader, epochs=100, lr=0.01, in_dim=in_size, latent_dim=512, device=device)
    target_graph_enc = ['PubMed', 'CiteSeer', 'Cora']
    graph_encodings, _ = get_graph_enc(target_graph_enc, device)
    for idx, dataset in enumerate(target_graph_enc):
        enc = graph_encodings[idx]
        enc = model.get_encoding(enc)
        path = f'./param_data/{dataset}/contrastive_encoding.pt'
        torch.save(enc.detach().cpu(), path)
        print(f'{dataset} saved!')
    # torch.save(model.param_encoder, './param_data/param_encoder.pt')

if __name__ == '__main__':
    main()