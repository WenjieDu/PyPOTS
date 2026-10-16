"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause
#
# Portions of this file are adapted from the following project(s), redistributed under
# their original license terms reproduced below:
#   - AROMA: https://github.com/LouisSerrano/aroma
#   - perceiver-pytorch: https://github.com/lucidrains/perceiver-pytorch
#
# ---- AROMA (https://github.com/LouisSerrano/aroma) ----
#
# MIT License
#
# Copyright (c) 2024 LouisSerrano
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- perceiver-pytorch (https://github.com/lucidrains/perceiver-pytorch) ----
#
# MIT License
#
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import einsum, nn

from ..utils import default, exists
from .utils import GEGLU


class FeedForward(nn.Module):
    """A two-layer MLP with GeLU or Gated GeLU intermediate activation."""

    def __init__(self, dim: int, mult: int = 4, use_geglu: bool = False) -> None:
        """
        A two-layer MLP with GeLU or Gated GeLU intermediate activation.

        Args:
            dim (int): input and output dim
            mult (int): multiplicative factor for the hidden layer dim
            use_geglu (bool): True if Gated GELU activation, False if GeLU act
        """

        super().__init__()
        if use_geglu:
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GEGLU(),
                nn.Linear(dim * mult, dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult),
                nn.GELU(),
                nn.Linear(dim * mult, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """An implementation of multi-head cross-attention with arbitrary q, k, v."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        headwise_attn_output_gate: bool = False,
    ) -> None:
        """
        Multi-head cross-attention with arbitrary q, k, v.

        Linear projection to Q and K, V -> one-layer attention -> feedforward.

        Args:
            query_dim (int): input dim of the query
            key_dim (int): input dim of the key
            value_dim (int): input dim of the values
            heads (int): number of heads
            dim_head (int): dim of one attention head
            dropout (float): dropout frac
        """

        super().__init__()

        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.inner_dim = inner_dim
        self.gating = headwise_attn_output_gate

        # input projectors:
        to_q_out_dim = inner_dim + int(heads * self.gating)
        self.to_q = nn.Linear(query_dim, to_q_out_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        # attention and residual connection dropouts:
        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        h = self.heads

        q = self.to_q(x)  # (bs, seq_len, d_model) with d_model = n_heads x dim_head
        if self.gating:
            q, gate_scores = torch.split(q, [self.inner_dim, self.heads], dim=-1)
            gate_scores = rearrange(gate_scores, "b n h -> b n h 1")

        # context = default(context, x)
        k = self.to_k(k)  # (bs, seq_len_kv, d_model)
        v = self.to_v(v)  # (bs, seq_len_kv, d_model)

        # sort per head:
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # compute per-head attention scores:
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale  # (bs x n_heads, seq_len, seq_len_kv)

        # apply attention mask, if need be:
        if exists(mask):
            mask = mask.bool()
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) (n) j", h=h, n=x.shape[1])
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of:
        half = sim.shape[0] // 2
        with torch.autocast(device_type=sim.device.type, enabled=False):
            sim[:half] = self.attn_drop(sim[:half].softmax(dim=-1))  # (bs x n_heads, seq_len, seq_len_kv)
            sim[half:] = self.attn_drop(sim[half:].softmax(dim=-1))  # (bs x n_heads, seq_len, seq_len_kv)

        out = einsum("b i j, b j d -> b i d", sim, v)  # (bs x n_heads, num_bandwidth, seq_len, dim_head)
        if self.gating:
            out = rearrange(out, "(b h) n d -> b n h d", h=h)  # (bs, seq_len, n_heads, dim_head)
            out = out * torch.sigmoid(gate_scores)  # (bs, seq_len, n_heads, dim_head)
            out = rearrange(out, "b n h d -> b n (h d)")  # (bs, seq_len, d_model)
        else:
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)  # (bs, seq_len, d_model)

        return self.resid_drop(self.to_out(out))  # (bs, seq_len, query_dim)


class MultiScaleAttention(nn.Module):
    """
    Multi-head multi-scale (queries at different frequency bandwidths) self-attention,
    or cross-attention if context is provided for KV tensors.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        headwise_attn_output_gate: bool = False,
    ) -> None:
        """
        Multi-head multi-scale (queries at different frequency bandwidths) self-attention,
        or cross-attention if context is provided for KV tensors.

        Linear projection to Q and K, V (jointly from context) -> one-layer attention -> feedforward.

        Args:
            query_dim (int): input dim of the query
            context_dim (int): input dim of the context, defaults to `query_dim`
            out_dim (int): dim of the output, defaults to `query_dim`
            heads (int): number of attention heads
            dim_head (int): dim of attention head
            dropout (float): attention and ff dropout
        """

        super().__init__()

        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.inner_dim = inner_dim
        self.gating = headwise_attn_output_gate

        # input projectors:
        to_q_out_dim = inner_dim + int(heads * self.gating)
        self.to_q = nn.Linear(query_dim, to_q_out_dim, bias=False)
        self.to_kv = nn.Linear(default(context_dim, query_dim), inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, default(out_dim, query_dim))

        # attention and residual connection dropouts:
        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        h = self.heads

        # get query:
        q = self.to_q(x)  # (bs, seq_len, num_bandwidth, d_model)
        if self.gating:
            q, gate_scores = torch.split(q, [self.inner_dim, self.heads], dim=-1)
            gate_scores = rearrange(gate_scores, "b n s h -> b n s h 1")

        # get kv:
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)  # (bs, seq_len_kv, d_model)
        k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k, v)
        )  # (bs x n_heads, seq_len_kv, dim_head)

        # the s stands for scale as we have queries at different frequency bandwidths:
        q = rearrange(q, "b n s (h d) -> (b h) s n d", h=h)  # (bs x n_heads, num_bandwidth, seq_len, dim_head)

        # compute attention product:
        sim = (
            einsum("b s i d, b j d -> b s i j", q, k) * self.scale
        )  # (bs x n_heads, num_bandwidth, seq_len, seq_len_kv)

        # optionally add attention mask:
        if exists(mask):
            mask = mask.bool()
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of:
        half = sim.shape[0] // 2
        with torch.autocast(device_type=sim.device.type, enabled=False):
            sim[:half] = self.attn_drop(
                sim[:half].softmax(dim=-1)
            )  # (bs x n_heads, num_bandwidth, seq_len, seq_len_kv)
            sim[half:] = self.attn_drop(
                sim[half:].softmax(dim=-1)
            )  # (bs x n_heads, num_bandwidth, seq_len, seq_len_kv)

        # end of attention product:
        out = einsum("b s i j, b j d -> b s i d", sim, v)  # (bs x n_heads, num_bandwidth, seq_len, dim_head)
        if self.gating:
            out = rearrange(out, "(b h) s n d -> b n s h d", h=h)  # (bs, seq_len, n_heads, dim_head)
            out = out * torch.sigmoid(gate_scores)  # (bs, seq_len, n_heads, dim_head)
            out = rearrange(out, "b n s h d -> b n s (h d)")  # (bs, seq_len, d_model)
        else:
            out = rearrange(out, "(b h) s n d -> b n s (h d)", h=h)  # (bs, seq_len, num_bandwidth, d_model)

        return self.resid_drop(self.to_out(out))  # (bs, seq_len, num_bandwidth, dim_query)


class Attention(nn.Module):
    """Multi-head self-attention, or cross-attention if context is provided for KV tensors."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        headwise_attn_output_gate: bool = False,
    ) -> None:
        """
        Multi-head self-attention or cross-attention if context is provided for KV tensors.

        Linear projection to Q and K, V (jointly from context) -> one-layer attention -> feedforward.

        Args:
            query_dim (int): input dim of the query
            context_dim (int): input dim of the context, defaults to `query_dim`
            out_dim (int): dim of the output, defaults to `query_dim`
            heads (int): number of attention heads
            dim_head (int): dim of attention head
            dropout (float): attention and ff dropout
        """

        super().__init__()

        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.inner_dim = inner_dim
        self.gating = headwise_attn_output_gate

        # input projectors:
        to_q_out_dim = inner_dim + int(heads * self.gating)
        self.to_q = nn.Linear(query_dim, to_q_out_dim, bias=False)
        self.to_kv = nn.Linear(default(context_dim, query_dim), inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, default(out_dim, query_dim))

        # attention and residual connection dropouts:
        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        h = self.heads

        # get query:
        q = self.to_q(x)  # (bs, seq_len, d_model)
        if self.gating:
            q, gate_scores = torch.split(q, [self.inner_dim, self.heads], dim=-1)
            gate_scores = rearrange(gate_scores, "b n h -> b n h 1")

        # get k,v:
        context = default(context, x)  # (bs, seq_len_kv, d_kv)
        k, v = self.to_kv(context).chunk(2, dim=-1)  # (bs, seq_len_kv, d_model)

        # rearrange per head:
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # compute dot-product attention:
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale  # (bs x n_heads, seq_len, seq_len_kv)

        # optionally add attention mask:
        if exists(mask):
            mask = mask.bool()
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention:
        half = sim.shape[0] // 2
        with torch.autocast(device_type=sim.device.type, enabled=False):
            sim[:half] = self.attn_drop(sim[:half].softmax(dim=-1))  # (bs x n_heads, seq_len, seq_len_kv)
            sim[half:] = self.attn_drop(sim[half:].softmax(dim=-1))  # (bs x n_heads, seq_len, seq_len_kv)

        # end of attention product:
        out = einsum("b i j, b j d -> b i d", sim, v)  # (bs x n_heads, seq_len, dim_head)
        if self.gating:
            out = rearrange(out, "(b h) n d -> b n h d", h=h)  # (bs, seq_len, n_heads, dim_head)
            out = out * torch.sigmoid(gate_scores)  # (bs, seq_len, n_heads, dim_head)
            out = rearrange(out, "b n h d -> b n (h d)")  # (bs, seq_len, d_model)
        else:
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)  # (bs, seq_len, d_model)

        return self.resid_drop(self.to_out(out))  # (bs, seq_len, dim_query)
