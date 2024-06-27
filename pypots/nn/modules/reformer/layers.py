"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .lsh_attention import LSHSelfAttention
from ..transformer import PositionWiseFeedForward


class ReformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        bucket_size,
        n_hashes,
        causal,
        d_ffn,
        dropout,
    ):
        super().__init__()
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        enc_input: torch.Tensor,
    ):
        enc_output = self.attn(enc_input)

        # apply dropout and residual connection
        enc_output = self.dropout(enc_output)
        enc_output += enc_input

        # apply layer-norm
        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output
