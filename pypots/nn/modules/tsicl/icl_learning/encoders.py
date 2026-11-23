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
# https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_model/encoders.py
from typing import Callable, Optional

from torch import Tensor, nn

from .layers import MultiheadAttentionBlock
from .rope import RotaryEmbedding


class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
        Number of multihead attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    use_rope : bool, default=False
        Whether to use rotary positional encoding

    rope_base : int, default=100000
        A base scaling factor for rotary position encoding
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        use_rope: bool = False,
        rope_base: int = 100000,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

        self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base) if use_rope else None

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
    ) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

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

        Returns
        -------
        Tensor
            Output tensor of shape (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(q=out, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=self.rope)

        return out
