"""Implementation of pypots.nn.modules.autoformer.autoencoder."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from torch import nn

from ..informer.autoencoder import InformerEncoder
from .layers import (
    AutoCorrelation,
    AutoformerEncoderLayer,
    SeasonalLayerNorm,
)


class AutoformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        factor,
        moving_avg_window_size,
        dropout,
        activation="relu",
    ):
        super().__init__()

        self.encoder = InformerEncoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelation(factor, dropout),
                    d_model,
                    n_heads,
                    d_ffn,
                    moving_avg_window_size,
                    dropout,
                    activation,
                )
                for _ in range(n_layers)
            ],
            norm_layer=SeasonalLayerNorm(d_model),
        )

    def forward(self, x, attn_mask=None):
        enc_out, attns = self.encoder(x, attn_mask)
        return enc_out, attns
