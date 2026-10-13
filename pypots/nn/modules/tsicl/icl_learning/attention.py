"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause
#
# Portions of this file are adapted from the following project(s), redistributed under
# their original license terms reproduced below:
#   - TabICL: https://github.com/soda-inria/tabicl
#
# ---- TabICL (https://github.com/soda-inria/tabicl) ----
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Soda team @ Inria
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

# ACKNOWLEDGEMENT:
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/attention.py
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .rope import RotaryEmbedding


def sdpa_with_flattened_batch(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's scaled_dot_product_attention and then reshaping the output
    back to the original shape. This flattening is necessary to properly trigger
    Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:
        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len)

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input
    """

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor | int] = None,
    rope: Optional[RotaryEmbedding] = None,
) -> Tensor:
    """Multi-head attention with support for rotary position embeddings
    as well as specialized processing when attn_mask is an integer.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim)

    key : Tensor
        Key tensor of shape (..., src_len, embed_dim)

    value : Tensor
        Value tensor of shape (..., src_len, embed_dim)

    num_heads : int
        Number of attention heads

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections

    in_proj_bias : Tensor
        Combined bias vector for input projections

    dropout_p : float
        Dropout probability applied to attention weights

    out_proj_weight : Tensor
        Output projection weight matrix

    out_proj_bias : Tensor
        Output projection bias vector

    training : bool, default=True
        Whether the model is in training mode (affects dropout)

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:
            - For binary masks: True values indicate positions to ignore
            - For float masks: Values are directly added to attention scores

    attn_mask : Optional[Tensor | int], default=None
        Controls attention pattern in two possible ways:
        1. When provided as Tensor: Traditional mask preventing attention to certain positions
            - Shape: (tgt_len, src_len) or (..., num_heads, tgt_len, src_len)
        2. When provided as integer: Creates a split attention pattern where:
            - The first `attn_mask` tokens perform self-attention only (attend to themselves)
            - The remaining tokens attend only to the first `attn_mask` tokens

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., tgt_len, embed_dim)
    """

    if isinstance(attn_mask, int):
        assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
        assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key.shape[-2]

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # Joint projection of query, key, value
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # Reshape for multi-head attention
    q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, tgt_len, hs)
    k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)
    v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)

    # Apply rotary position embeddings if provided
    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    if isinstance(attn_mask, int):
        cut_pos = attn_mask  # For better readability

        # Pre-allocate output tensor to avoid concatenation
        attn_output = torch.empty(*batch_shape, tgt_len, embed_dim, device=query.device, dtype=query.dtype)

        # Process left segment (self-attention within first cut_pos tokens)
        q_left = q[..., :cut_pos, :]  # (batch_shape, nh, cut_pos, hs)
        k_left = k[..., :cut_pos, :]
        v_left = v[..., :cut_pos, :]

        attn_left = sdpa_with_flattened_batch(q_left, k_left, v_left, dropout_p=dropout_p)
        attn_left = attn_left.transpose(-3, -2).contiguous().view(*batch_shape, cut_pos, embed_dim)
        attn_output[..., :cut_pos, :] = F.linear(attn_left, out_proj_weight, out_proj_bias)

        # Process right segment (tokens after cut_pos attending to tokens before cut_pos)
        if cut_pos < tgt_len:
            q_right = q[..., cut_pos:, :]  # (batch_shape, nh, tgt_len - cut_pos, hs)
            attn_right = sdpa_with_flattened_batch(q_right, k_left, v_left, dropout_p=dropout_p)
            attn_right = attn_right.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len - cut_pos, embed_dim)
            attn_output[..., cut_pos:, :] = F.linear(attn_right, out_proj_weight, out_proj_bias)
    else:
        # Process attention mask
        correct_2d_shape = (tgt_len, src_len)
        correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != correct_2d_shape:
                    raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
                attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
            elif attn_mask.dim() == len(correct_nd_shape):
                if attn_mask.shape != correct_nd_shape:
                    raise ValueError(
                        f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                        f"but got {attn_mask.shape}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

        # Process key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (*batch_shape, src_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
                *batch_shape, num_heads, tgt_len, src_len
            )

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        attn_output = sdpa_with_flattened_batch(q, k, v, attn_mask, dropout_p)  # (..., nh, tgt_len, hs)

        # Reshape and project output
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    return attn_output
