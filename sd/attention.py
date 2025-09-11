import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x : (Batch_size, Seq_len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) ->  3 tensors of shape (batch_size, seq_len, dim)
        (
            q,
            k,
            v,
        ) = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, Dim) -> (batch_size, seq_len, H, Dim/H) -> (Batch_size, H, seq_len, Dim/H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask where the upper triange above the upper diagonal is ones
            mask = torch.ones_like(weight)
            # weight.masked_fill_(mask, -torch.inf)
            weight.masked_fill_(mask.to(torch.bool), -torch.inf)

        # Scale and softmax
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, Dim/H) -> (batch_size, h, seq_len, dim/H)
        # apply attention weights to v
        output = weight @ v

        # (batch_size, h, seq_len, dim/h) -> (batch_size, seq_len, H, dim/h)
        # combine heads
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (batch_size, seq_len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, d_cross, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x, y):
        # x : (latent) of shape (batch_size, seq_len_Q, Dim_Q)
        # y : (context) of shape (batch_size, seq_len_KV, Dim_KV) = (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_heads)

        # Multiplyquery by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output
