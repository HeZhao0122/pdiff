import numpy as np
import math

import torch
from torch import nn
import random
import torch.nn.functional as F

def param_statistics(model):
    params = 0
    for module, weights in model.named_parameters():
        weights = weights.data.view(-1)
        params += weights.shape[0]
    return params

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm

class TFVae(nn.Module):
    def __init__(
            self,
            in_dim,
            dim_list,
            latent_dim,
            len_token,
            input_noise_factor,
            latent_noise_factor,
            num_layers=3,
            param_layer=2,
            kld_weight=0.01
    ):
        super().__init__()
        self.dim_list = dim_list
        self.len_token = len_token
        self.kld_weight = kld_weight
        # self.in_dim = (in_dim//self.len_token+1)*self.len_token
        # self.real_dim = in_dim
        self.latent_dim = latent_dim
        # self.dropout = nn.Dropout(p=dropout)
        self.lin_enc = nn.ModuleList()
        self.lin_dec = nn.ModuleList()
        for dim in dim_list:
            self.lin_enc.append(nn.Linear(dim, self.len_token))
            self.lin_dec.append(nn.Linear(self.len_token, dim))

        self.encoder = TransformerEncoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=self.latent_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        ff_dim = len(dim_list) * self.len_token
        self.norm = Norm(in_dim)
        self.de_norm = Norm(in_dim)
        self.fc_mu = TransformerEncoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=self.latent_dim,
            num_layers=1,
            dropout=0.1,
        )
        self.fc_var = TransformerEncoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=self.latent_dim,
            num_layers=1,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=self.latent_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        self.pos_emb = ParamPosEmbedding(self.len_token, len(dim_list))
        # self.pos_emb = LayerPosEmbedding(self.len_token, param_layer, len(dim_list))

    def forward(self, input, train=None):
        # input = F.normalize(input, dim=1, p=2)
        # tf_input = self.add_noise(input, self.input_noise_factor)
        latent = self.encode(input)
        # latent_flattened = latent.view(latent.size(0), -1)
        # print('time_emb.shape', time_emb.shape)
        # print('emb_enc1.shape', emb_enc1.shape)
        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        z = self.reparameterize(mu, log_var)
        # z = z.view(latent.shape)
        # z = self.add_noise(z, self.latent_noise_factor)
        # dec_latent = torch.clamp(z, -1, 1)
        out = self.decode(z)

        recons = out
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_loss = torch.mean(kld_loss)

        loss = recons_loss + self.kld_weight * kld_loss
        # return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
        return loss

    def reconstruct(self, input):
        latent = self.encode(input)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        z = self.reparameterize(mu, log_var)

        out = self.decode(z)
        return latent, out

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.dec_channel_list[0], self.dec_dim_list[0]).cuda()
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        z = self.reparameterize(mu, log_var)
        samples = self.decode(z)
        return samples

    def encode(self, input):
        # input = self.norm(input)
        input = input.unsqueeze(1)
        tf_input = []
        pos = 0
        for dim, enc in zip(self.dim_list, self.lin_enc):
            tf_input.append(enc(input[:, :, pos:pos + dim]))
            pos += dim
        tf_input = torch.cat(tf_input, dim=1)
        tf_input = F.tanh(tf_input)
        tf_input = tf_input + self.pos_emb(tf_input)
        # tf_input = self.dropout(tf_input)

        # input_seq = tf_input.reshape(input_view_shape[0], -1, self.len_token)
        latent = self.encoder(tf_input)
        # out = self.enc_ff(latent)

        # import pdb; pdb.set_trace()
        # time_emb = self.time_encode(time)
        # time_emb_rs = time_emb.reshape(input_view_shape[0], 1, self.len_token)
        return latent

    def decode(self, latent):
        # latent = self.dec_ff(latent)
        latent = latent + self.pos_emb(latent)
        tf_out = self.decoder(latent, latent)
        tf_out = F.tanh(tf_out)
        tf_out = torch.chunk(tf_out, len(self.dim_list), dim=1)
        tf_out = [t.squeeze(1) for t in tf_out]
        out = []
        for emb, dec in zip(tf_out, self.lin_dec):
            out.append(dec(emb))
        out = torch.cat(out, dim=1)
        # out = self.de_norm(out)
        # out = out[:, : self.real_dim]
        # out = out.reshape((latent.shape[0], self.real_dim))
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, layer):
        super(MLP, self).__init__()
        self.dim_list = np.linspace(in_dim, hidden_size, layer+1)
        self.dim_list = [int(i) for i in self.dim_list]
        self.enc = nn.ModuleList()
        for i in range(layer):
            l = nn.Sequential(nn.Linear(self.dim_list[i], self.dim_list[i+1]),
                              nn.Tanh())
            self.enc.append(l)

    def forward(self, x):
        for l in self.enc:
            x = l(x)
        return x


class TF(nn.Module):
    def __init__(
            self,
            in_dim,
            dim_list,
            latent_dim,
            len_token,
            input_noise_factor,
            latent_noise_factor,
            num_layers=3,
            param_layer=2,
    ):
        super().__init__()
        self.dim_list = dim_list
        self.len_token = len_token
        self.loss_func = nn.MSELoss()
        # self.in_dim = (in_dim//self.len_token+1)*self.len_token
        # self.real_dim = in_dim
        self.latent_dim = latent_dim
        # self.dropout = nn.Dropout(p=dropout)
        self.lin_enc = nn.ModuleList()
        self.lin_dec = nn.ModuleList()
        # for dim in dim_list:
        #     self.lin_enc.append(nn.Linear(dim, self.len_token))
        #     self.lin_dec.append(nn.Linear(self.len_token, dim))
        for dim in dim_list:
            self.lin_enc.append(MLP(dim, self.len_token, 1))
            self.lin_dec.append(MLP(self.len_token, dim, 1))

        self.encoder = TransformerEncoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=self.latent_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=self.latent_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        self.pos_emb = ParamPosEmbedding(self.len_token, len(dim_list))
        # self.pos_emb = LayerPosEmbedding(self.len_token, param_layer, len(dim_list))

    def forward(self, input, train=None):
        tf_input = self.add_noise(input, self.input_noise_factor)
        latent = self.encode(tf_input)
        # print('time_emb.shape', time_emb.shape)
        # print('emb_enc1.shape', emb_enc1.shape)
        latent = self.add_noise(latent, self.latent_noise_factor)
        latent = torch.clamp(latent, -1, 1)
        out = self.decode(latent)

        # scaled MSE
        # loss = self.scaledMSE(out, input)
        loss = self.loss_func(out, input)
        # if input_view_shape[1] < self.in_dim:
        return loss

    def scaledMSE(self, x, input):
        pos = 0
        loss = 0.0
        for dim in self.dim_list:
            var = torch.var(input[:, pos: pos+dim], dim=1)
            loss = loss + self.loss_func(x[:, pos:pos+dim], input[:, pos: pos+dim])/var
            pos = pos + dim
        loss = loss.mean()/len(self.dim_list)
        return loss

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def encode(self, input):
        input = input.unsqueeze(1)
        tf_input = []
        pos = 0
        for dim, enc in zip(self.dim_list, self.lin_enc):
            tf_input.append(enc(input[:, :, pos:pos + dim]))
            pos += dim
        tf_input = torch.cat(tf_input, dim=1)
        # tf_input = F.tanh(tf_input)
        tf_input = tf_input + self.pos_emb(tf_input)
        # tf_input = self.dropout(tf_input)

        # input_seq = tf_input.reshape(input_view_shape[0], -1, self.len_token)
        latent = self.encoder(tf_input)
        # out = self.enc_ff(latent)

        # import pdb; pdb.set_trace()
        # time_emb = self.time_encode(time)
        # time_emb_rs = time_emb.reshape(input_view_shape[0], 1, self.len_token)
        return latent

    def decode(self, latent):
        # latent = self.dec_ff(latent)
        latent = latent + self.pos_emb(latent)
        tf_out = self.decoder(latent, latent)
        # tf_out = F.tanh(tf_out)
        tf_out = torch.chunk(tf_out, len(self.dim_list), dim=1)
        tf_out = [t.squeeze(1) for t in tf_out]
        out = []
        for emb, dec in zip(tf_out, self.lin_dec):
            out.append(dec(emb))
        out = torch.cat(out, dim=1)
        # out = out[:, : self.real_dim]
        # out = out.reshape((latent.shape[0], self.real_dim))
        return out


class NewTF(nn.Module):
    def __init__(
            self,
            in_dim,
            len_token,
            ff_dim,
            latent_dim,
            input_noise_factor,
            latent_noise_factor,
            num_layers=3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.len_token = len_token
        self.loss_func = nn.MSELoss()
        self.latent_dim = latent_dim
        # self.dropout = nn.Dropout(p=dropout)
        self.token_embeddings = Embedder(in_dim, self.len_token)
        self.token_debeddings = Debedder(in_dim, self.len_token)

        self.encoder = TransformerEncoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=ff_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.vec2neck = nn.Sequential(nn.Linear(in_dim*self.len_token, latent_dim), nn.Tanh())
        self.neck2vec = nn.Sequential(nn.Linear(latent_dim, in_dim*self.len_token), nn.Tanh())
        self.decoder = TransformerDecoder(
            d_model=self.len_token,
            nhead=4,
            dim_feedforward=ff_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        # print(param_statistics(self.token_debeddings))
        self.pos_emb = PositionalEncoding(self.len_token, max_len=in_dim)
        # self.pos_emb = LayerPosEmbedding(self.len_token, param_layer, len(dim_list))

    def forward(self, input, train=None):
        # tf_input = self.add_noise(input, self.input_noise_factor)
        latent = self.encode(input)
        # print('time_emb.shape', time_emb.shape)
        # print('emb_enc1.shape', emb_enc1.shape)
        # latent = self.add_noise(latent, self.latent_noise_factor)
        # latent = torch.clamp(latent, -1, 1)
        out = self.decode(latent)

        # scaled MSE
        # loss = self.scaledMSE(out, input)
        loss = self.loss_func(out, input)
        # if input_view_shape[1] < self.in_dim:
        return loss

    def scaledMSE(self, x, input):
        pos = 0
        loss = 0.0
        for dim in self.dim_list:
            var = torch.var(input[:, pos: pos + dim], dim=1)
            loss = loss + self.loss_func(x[:, pos:pos + dim], input[:, pos: pos + dim]) / var
            pos = pos + dim
        loss = loss.mean() / len(self.dim_list)
        return loss

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def encode(self, input):
        x = self.token_embeddings(input)
        # tf_input = []
        # pos = 0
        # for dim, enc in zip(self.dim_list, self.lin_enc):
        #     tf_input.append(enc(input[:, :, pos:pos + dim]))
        #     pos += dim
        # tf_input = torch.cat(tf_input, dim=1)
        # tf_input = F.tanh(tf_input)
        x = x + self.pos_emb(x)
        # tf_input = self.dropout(tf_input)

        # input_seq = tf_input.reshape(input_view_shape[0], -1, self.len_token)
        latent = self.encoder(x)
        latent = latent.view(latent.shape[0], -1)
        latent = self.vec2neck(latent)

        # out = self.enc_ff(latent)

        # import pdb; pdb.set_trace()
        # time_emb = self.time_encode(time)
        # time_emb_rs = time_emb.reshape(input_view_shape[0], 1, self.len_token)
        return latent

    def decode(self, latent):
        # latent = self.dec_ff(latent)
        latent = self.neck2vec(latent)
        latent = latent.view(latent.shape[0], self.in_dim, self.len_token)
        latent = latent + self.pos_emb(latent)
        y = self.decoder(latent, latent)
        y = self.token_debeddings(y)
        # out = out[:, : self.real_dim]
        # out = out.reshape((latent.shape[0], self.real_dim))
        return y



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LayerPosEmbedding(nn.Module):
    def __init__(self, len_token, num_layers, num_params):
        super(LayerPosEmbedding, self).__init__()
        self.num_layers = num_layers
        self.size = int(num_params/num_layers)
        self.pos_enc = [nn.Parameter(torch.randn(1, len_token)) for i in range(num_layers)]

    def forward(self, x):
        emb = []
        for layer in range(self.num_layers):
            for i in range(self.size):
                emb.append(self.pos_enc[layer])
        emb = torch.cat(emb, dim=0).to(x.device)
        return emb


class ParamPosEmbedding(nn.Module):
    def __init__(self, len_token, num_pos):
        super(ParamPosEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.randn((num_pos, len_token)))

    def forward(self, x):
        return self.embedding


class TransformerEncoder(nn.Module):
    def __init__(self, nhead, dim_feedforward, num_layers, d_model, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # self.pos_encoder = ParamPosEmbedding(d_model)

    def forward(self, src):
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, nhead, dim_feedforward, num_layers, d_model, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        # self.pos_decoder = PositionalEncoding(d_model)

    def forward(self, tgt, memory):
        # tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output


class Embedder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(1, embed_dim)

    def forward(self, x):
        y = []
        # use the same embedder to embedd all weights
        for idx in range(self.input_dim):
            # embedd single input / feature dimension
            tmp = self.embed(x[:, idx].unsqueeze(dim=1))
            y.append(tmp)
        # stack along dimension 1
        y = torch.stack(y, dim=1)
        return y


class Debedder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        # self.input_dim = input_dim
        self.d_model = d_model
        self.weight_debedder = nn.Linear(d_model, 1)

    def forward(self, x):
        y = self.weight_debedder(x)
        y = y.squeeze()
        return y



