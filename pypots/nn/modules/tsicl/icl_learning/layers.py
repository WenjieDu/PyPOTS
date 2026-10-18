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
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/layers.py
from typing import Callable, Optional, Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from .attention import multi_head_attention_forward
from .rope import RotaryEmbedding


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Parameters
    ----------
    in_dim : int
        Input feature dimension

    out_dim : Optional[int], default=None
        Output dimension. If None, uses the last hidden dimension

    hidden_dims : Sequence[int], default=(256, 256, 256)
        Dimensions of hidden layers

    activation : str, default='gelu'
        Activation function: 'relu', 'gelu', 'leaky_relu', or 'tanh'

    bias : bool, default=True
        Whether to include bias terms in linear layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        # Build network architecture
        act_fn = self.get_activation(activation)
        layers = []

        # Create hidden layers with activations
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(act_fn())
            prev_dim = hidden_dim

        # Optional output projection
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def get_activation(activation: str) -> nn.Module:
        """Get activation function class from string name.

        Parameters
        ----------
        activation : str
            Name of activation function

        Returns
        -------
        class
            PyTorch activation function class
        """

        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Supported: {list(activation_map.keys())}")

        return activation_map[activation]

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (..., in_dim)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_dim or last_hidden_dim)
        """
        return self.net(X)


class MultiheadAttention(nn.MultiheadAttention):
    """Enhanced multi-head attention with rotary positional embedding support.

    This extends PyTorch's MultiheadAttention to support rotary position embeddings (RoPE)
    and specialized attention masking when `attn_mask` is an integer. The implementation always
    uses `batch_first=True`, meaning all input tensors have shape (..., seq_len, embed_dim).

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined)

    num_heads : int
        Number of attention heads

    dropout : float, default=0.0
        Dropout probability applied to attention weights

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Compute multi-head attention with support for rotary positional encoding.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (..., tgt_len, embed_dim)

        key : Tensor
            Key tensor of shape (..., src_len, embed_dim)

        value : Tensor
            Value tensor of shape (..., src_len, embed_dim)

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
            Attention output of shape (..., tgt_len, embed_dim)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

        return multi_head_attention_forward(
            query,
            key,
            value,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
        )


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    """Attention block supporting rotary positional encoding.

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
       Dimension of the feedforward network

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first=norm_first, batch_first=True)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize projection layers to zero for stable training."""
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Process input through attention with optional rotary positional encoding.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (..., tgt_len, d_model)

        k : Tensor
            Key tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

        v : Tensor
            Value tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

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
            Output tensor of shape (..., tgt_len, d_model)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"
        else:
            # Convert masks to correct dtype for compatibility
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="src_mask",
                target_type=q.dtype,
            )
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=q.dtype,
                check_other=False,
            )

        # Use q as k,v if not provided
        # k = q if k is None else k
        # v = q if v is None else v

        # Apply layer depending on normalization order
        x = q
        if self.norm_first:
            # Pre-norm: normalize before attention and FFN
            attn = self._attn_block(
                self.norm1(q),
                self.norm1(q if k is None else k),
                self.norm1(q if v is None else v),
                key_padding_mask,
                attn_mask,
                rope,
            )
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm: normalize after attention and FFN
            attn = self._attn_block(q, q if k is None else k, q if v is None else v, key_padding_mask, attn_mask, rope)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor | int],
        rope: Optional[RotaryEmbedding],
    ) -> Tensor:
        attn = self.attn(q, k, v, key_padding_mask, attn_mask, rope)
        return self.dropout1(attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
