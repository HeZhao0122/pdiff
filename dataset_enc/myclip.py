import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
from set_transformer.models import SetTransformer
from tqdm import  tqdm
from set_transformer.super_linear import LinearSuper
from graph_encoding import get_graph_encoding

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLPEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU()
        )
    def forward(self, input):
        # out = []
        # for i in input:
        #     out.append(torch.mean(self.mlp(i), dim=0).view(1,-1))
        # return torch.cat(out, dim=0).unsqueeze(1)
        return torch.mean(self.mlp(input), dim=1)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_inds, hidden_dim, num_heads):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.intra = MLPEncoder(in_dim, in_dim * 4, in_dim)
        self.inter = MLPEncoder(in_dim, in_dim * 4, in_dim)
        # self.intra = SetTransformer(in_dim, num_inds, hidden_dim, num_heads)
        # self.inter = SetTransformer(in_dim, num_inds, hidden_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            if isinstance(x, list) and len(x) > 1:
                x = torch.stack(x, dim=0)
            elif isinstance(x, list) and len(x) == 1:
                x = x[0]

            x = x.cuda()
            z = self.intra(x).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)
        outputs = torch.stack(outputs, 0)
        outputs = self.proj(outputs).reshape(-1, self.out_dim)
        return outputs


class EmbedData(nn.Module):
    def __init__(self,  in_dim, feature_size, num_inds=32, hidden_dim=128, num_heads=2, emb_dim=1024, use_feat=False, **kwargs):
        super(EmbedData, self).__init__()
        if feature_size is not None:
            self.feature_encoder = Encoder(feature_size, in_dim, num_inds, hidden_dim, num_heads)
        self.structure_encoder = Encoder(in_dim, in_dim, num_inds, hidden_dim, num_heads)
        # self.intra = SetTransformer(in_dim, num_inds, hidden_dim, num_heads)
        # self.inter = SetTransformer(in_dim, num_inds, hidden_dim, num_heads)
        # self.intra = MLPEncoder(in_dim, in_dim*4, in_dim)
        # self.inter = MLPEncoder(in_dim, in_dim*4, in_dim)
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        # self.input_size= input_size
        # self.channels = channels
        if use_feat:
            self.proj = nn.Linear(in_dim * 2, emb_dim)
        else:
            self.proj = nn.Linear(in_dim, emb_dim)

    def forward(self, struc, feats):
        # struc, feats = inputs
        struc_emb = self.structure_encoder(struc)
        if feats is not None:
            feats = self.feature_encoder(feats)
            outputs = torch.cat([struc_emb,feats], dim=-1)
        else:
            outputs = struc_emb
        outputs = self.proj(outputs)
        return outputs


class CLIP(nn.Module):
    def __init__(self, in_dim, feat_size, num_inds=32, hidden_dim=128, num_heads=2, out_dim=512, temp=1.0, device='cuda', use_feat=False):
        super(CLIP, self).__init__()
        self.dataset_encoder = EmbedData(in_dim,feat_size, num_inds, hidden_dim, num_heads, emb_dim=out_dim, use_feat=use_feat)
        # self.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)
        self.temp = temp
        self.flag = use_feat

    def forward(self, inputs,):
        if self.flag:
            weight_embedding, struc_embedding, feat_embedding = inputs
            dataset_embedding = self.dataset_encoder(struc_embedding, feat_embedding)
        else:
            weight_embedding, struc_embedding, _ = inputs
            dataset_embedding = self.dataset_encoder(struc_embedding, None)

        weight_embedding = F.normalize(weight_embedding, p=2, dim=1)
        dataset_embedding = F.normalize(dataset_embedding, p=2, dim=1)

        # logits = (weight_embedding @ dataset_embedding.T)/self.temp
        # # dataset_similarity = dataset_embedding @ dataset_embedding.T
        # # weight_similarity = weight_embedding @ weight_embedding.T
        #
        # # targets = F.softmax(
        # #     (dataset_similarity + weight_similarity) / 2 * self.temp, dim=-1)
        # labels = torch.arange(weight_embedding.shape[0], device=weight_embedding.device)
        # weight_loss = F.cross_entropy(logits, labels)
        # dataset_loss = F.cross_entropy(logits.T, labels)
        # loss = (dataset_loss + weight_loss) / 2.0
        logits = (weight_embedding @ dataset_embedding.T) / self.temp
        dataset_similarity = dataset_embedding @ dataset_embedding.T
        weight_similarity = weight_embedding @ weight_embedding.T
        targets = F.softmax(
            (dataset_similarity + weight_similarity) / 2 * self.temp, dim=-1
        )
        weight_loss = cross_entropy(logits, targets, reduction='none')
        dataset_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (dataset_loss + weight_loss) / 2.0  # + F.mse_loss(dataset_embeddings, weight_embeddings)# shape: (batch_size)

        return loss.mean()

    def get_encoding(self, struc, feats):
        return self.dataset_encoder(struc, feats)


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def load_latent(root, device='cpu'):
    latents = torch.load(os.path.join(root, 'latent_cite_cora.pt'), map_location=device)
    latents = latents.view(latents.shape[0], -1)
    labels = torch.load(os.path.join(root, 'labels_cite_cora.pt'), map_location=device)
    dataset_num = int(torch.max(labels)) + 1
    res = [[] for _ in range(dataset_num)]
    for idx, l in enumerate(labels):
        res[int(l)].append(latents[idx].view(1, -1))
    print('finish loading!')
    return res

def padding(encoding, in_sizes, device):
    padding_length = [max(in_sizes) - i for i in in_sizes]
    new_enc = {}
    for idx, enc in encoding.items():
        new_e = []
        for e in enc:
            new_e.append(torch.cat((e.to(device), torch.zeros((e.shape[0], padding_length[idx]), device=device)),
                                   dim=-1))
        new_enc[idx] = new_e
    return new_enc


def get_graph_enc(dataset_list, device='cpu'):
    # ToDO padding 咋解决
    encoding = {}
    in_sizes = []
    feat_encoding = {}
    feat_in_sizes = []
    have_feat = False
    for idx, dataset in enumerate(dataset_list):
        enc, in_size, feat, feat_size = get_graph_encoding(dataset, device)
        encoding[idx] = enc
        in_sizes.append(in_size)
        if feat is not None:
            have_feat = True
            feat_encoding[idx] = feat
            feat_in_sizes.append(feat_size)
    struc_enc = padding(encoding, in_sizes, device)
    if have_feat:
        feat_enc = padding(feat_encoding, feat_in_sizes, device)
        return struc_enc, max(in_sizes), feat_enc, max(feat_in_sizes)
    else:
        return struc_enc, max(in_sizes), None, None


def get_pos_pair(graph_encoding, features, latents, num=200):
    pos_pair = []
    for idx, enc in graph_encoding.items():
        pos = latents[idx]
        for i in range(len(pos)):
            if features is not None:
                pos_pair.append([pos[i], graph_encoding[idx], features[idx]])
            else:
                pos_pair.append([pos[i], graph_encoding[idx], None])
    random.shuffle(pos_pair)
    return pos_pair

def generate_dataloader(pairs, batch_size=50):
    dataloader = []
    for idx in range(0,len(pairs), batch_size):
        data = pairs[idx: idx+batch_size]
        weight_embedding = []
        struc_embedding = []
        feat_embedding = []
        for p in data:
            weight_embedding.append(p[0])
            struc_embedding.append(p[1])
            feat_embedding.append(p[2])
        weight_embedding = torch.cat(weight_embedding, dim=0)
        dataloader.append([weight_embedding, struc_embedding, feat_embedding])
    return dataloader


def train(dataloder, epochs, lr, in_dim, feat_size,  num_inds=32, hidden_dim=128, num_heads=2, out_dim=512, temp=0.7, device='cuda', use_feat=False):
    model = CLIP(in_dim, feat_size, num_inds, hidden_dim, num_heads, out_dim, temp, device, use_feat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        num = 0
        for inputs in dataloder:
            num += 1
            optimizer.zero_grad()
            loss = model(inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 25 == 0:
            print(f'Epoch {epoch} loss: {epoch_loss/num}')
    print('Training Done!')
    return model


def main():
    device = 'cuda'
    latents = load_latent('.././param_data/', device)
    encodings, in_size, feats, feat_size = get_graph_enc(['CiteSeer', 'Cora'], device)
    pos_pair = get_pos_pair(encodings, feats, latents)
    dataloader = generate_dataloader(pos_pair, batch_size=50)
    model = train(dataloader, epochs=100, lr=1e-4, in_dim=in_size, feat_size=feat_size,
                  num_inds=32, hidden_dim=1024, num_heads=2, out_dim=32*32, temp=1.0, device=device, use_feat=False)
    target_graph_enc = ['PubMed', 'CiteSeer', 'Cora']
    graph_encodings, _, feats, _ = get_graph_enc(target_graph_enc, device)
    for idx, dataset in enumerate(target_graph_enc):
        enc = graph_encodings[idx]
        if feats is not None:
            feat = [feats[idx]]
        else:
            feat = None
        enc = model.get_encoding([enc], feat)
        path = f'../param_data/{dataset}/contrastive_encoding_identity.pt'
        torch.save(enc.detach().cpu(), path)
        print(f'{dataset} saved!')
if __name__ == '__main__':
    main()