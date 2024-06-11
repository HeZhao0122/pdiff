import math
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

# from model.x_transformer import AbsolutePositionalEmbedding


def exists(x):
    return x is not None

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def divisible_by(numer, denom):
    return (numer % denom) == 0


# NN components
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb



class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


def FeedForward(dim, mult=4, dropout=0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        qk_norm=True,
    ):
        super().__init__()
        hidden_dim = dim
        heads = dim // dim_head
        assert divisible_by(dim, heads), 'dimension must be divisible by number of heads'


        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = nn.LayerNorm(dim)

        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x,
    ):
        h = self.heads

        x = self.norm(x)


        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = einsum('b h i d, b h j d -> b h i j', self.query_norm(q)* self.scale, self.key_norm(k))

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_latent,
            dim_head=64,
            qk_norm=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        inner_dim = max(dim_latent, dim)
        self.heads = inner_dim // dim_head

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim_latent)

        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()

        self.to_q = nn.Linear(dim_latent, inner_dim, bias=False)
        if dim_latent != dim:
            self.latent_to_kv = nn.Linear(dim_latent, inner_dim * 2, bias=False)
        else:
            self.latent_to_kv = None
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_latent),
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)
        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if exists(self.latent_to_kv):
            kv_input = torch.cat([self.to_kv(x), self.latent_to_kv(latents)], dim=1)
        else:
            kv_input = torch.cat([self.to_kv(x), self.to_kv(latents)], dim=1)
        k, v = rearrange(kv_input, 'b n (split d) -> split b n d', split=2)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j',
                     self.query_norm(q) * self.scale, self.key_norm(k))

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_latent,
            depth,
            dim_head=64,
            num_latents=16,
            max_seq_len=64,
            ff_mult=4,
            legacy=False,
            l2_normalize_latents=False,
    ):
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        if legacy:
            dim_out = dim_latent
            dim_latent = dim

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(
                    dim=dim, dim_latent=dim_latent, dim_head=dim_head),
                FeedForward(dim=dim_latent, mult=ff_mult)
            ]))

        self.l2_normalize_latents = l2_normalize_latents

        self.final_norm = nn.LayerNorm(dim_latent)
        self.output_proj = nn.Linear(dim_latent, dim_out) if legacy else nn.Identity()

    def forward(self, x, mask=None):
        pos_emb = self.pos_emb(x)

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents
        latents = self.output_proj(self.final_norm(latents))
        # Normalize latents to norm sqrt(d_latent)
        if self.l2_normalize_latents:
            latents = F.normalize(latents, dim=-1) * math.sqrt(latents.shape[-1])
        return latents


class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim_input,
            dim_tx,
            depth,
            dim_head=64,
            max_seq_len=64,
            ff_mult=4,
    ):
        super().__init__()
        # self.pos_emb = AbsolutePositionalEmbedding(dim_tx, max_seq_len)

        self.input_proj = nn.Linear(dim_input, dim_tx)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim=dim_tx, dim_head=dim_head),
                FeedForward(dim=dim_tx, mult=ff_mult)
            ]))

        self.final_norm = nn.LayerNorm(dim_tx)
        self.output_proj = nn.Identity()

    def forward(self, x, mask=None):

        assert not exists(mask)
        x = self.input_proj(x)
        # pos_emb = self.pos_emb(x)
        x = x

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.output_proj(self.final_norm(x))


class PerceiverAutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            dim_lm,
            dim_ae,
            depth,
            dim_head=64,
            num_encoder_latents=8,
            num_decoder_latents=32,
            max_seq_len=64,
            ff_mult=4,
            encoder_only=False,
            transformer_decoder=False,
            l2_normalize_latents=False,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        if self.encoder_only:
            assert dim_ae == dim_lm
        # self.norm1 = nn.LayerNorm(dim_lm, elementwise_affine=False, eps=1e-6)
        # self.norm2 = nn.LayerNorm(dim_lm, elementwise_affine=False, eps=1e-6)
        self.perceiver_encoder = PerceiverResampler(dim=dim_lm, dim_latent=dim_ae, depth=depth, dim_head=dim_head,
                                                    num_latents=num_encoder_latents, max_seq_len=max_seq_len,
                                                    ff_mult=ff_mult, l2_normalize_latents=l2_normalize_latents)
        if transformer_decoder:
            self.perceiver_decoder = Transformer(dim_input=dim_ae, dim_tx=dim_lm, depth=depth, dim_head=dim_head,
                                                 max_seq_len=num_encoder_latents, ff_mult=ff_mult)
        else:
            self.perceiver_decoder = PerceiverResampler(dim=dim_ae, dim_latent=dim_lm, depth=depth, dim_head=dim_head,
                                                        num_latents=num_decoder_latents,
                                                        max_seq_len=num_encoder_latents, ff_mult=ff_mult)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, ae_latent):
        out = self.perceiver_decoder(ae_latent)
        return out

    def encode(self, encoder_outputs, attention_mask=None):
        # encoder_outputs = self.norm1(encoder_outputs)
        return self.perceiver_encoder(encoder_outputs, mask=attention_mask)

    def forward(self, encoder_input, attention_mask=None):
        inputs, mask = encoder_input
        # shape_info = mask.float().reshape(inputs.shape)
        # batch_size = inputs.shape[0]
        encoder_latents = self.perceiver_encoder(
            inputs, mask=attention_mask)
        # shape_latents = self.perceiver_encoder(shape_info, mask=attention_mask)
        # encoder_latents = encoder_latents.view(batch_size, -1)

        out = self.perceiver_decoder(encoder_latents)

        # recons = out
        # recons_loss = F.mse_loss(recons, inputs)
        #
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)
        #
        # loss = recons_loss + self.kld_weight * kld_loss
        # inputs = inputs.view(inputs.shape[0], -1)
        out = out.view(out.shape[0], -1)
        inputs = inputs.view(out.shape[0], -1)
        loss = F.mse_loss(out[mask], inputs[mask])
        return loss