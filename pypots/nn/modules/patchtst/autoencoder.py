"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ....nn.modules.transformer import TransformerEncoder


class PatchtstEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.encoder = TransformerEncoder(
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )

    def forward(self, x, attn_mask=None):

        enc_out, attns = self.encoder(x, attn_mask)

        enc_out = enc_out.reshape(-1, self.d_model, enc_out.shape[-2], enc_out.shape[-1])
        # [bz, d_model, d_model, n_patches] ->  [bz, d_model, n_patches, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)
        return enc_out, attns
