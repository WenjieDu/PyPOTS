"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .layers import (
    MultiWaveletTransform,
    MultiWaveletCross,
    FourierBlock,
    FourierCrossAttention,
)
from ....nn.modules.autoformer import (
    AutoformerEncoderLayer,
    AutoformerDecoderLayer,
    SeasonalLayerNorm,
)
from ....nn.modules.informer import InformerEncoder, InformerDecoder


class FEDformerEncoder(nn.Module):
    def __init__(
        self,
        n_steps,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        moving_avg_window_size,
        dropout,
        version="Fourier",
        modes=32,
        mode_select="random",
        activation="relu",
    ):
        super().__init__()

        if version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
        elif version == "Fourier":
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=n_steps,
                modes=modes,
                mode_select_method=mode_select,
            )
        else:
            raise ValueError(f"Unsupported version: {version}. Please choose from ['Wavelets', 'Fourier'].")

        self.encoder = InformerEncoder(
            [
                AutoformerEncoderLayer(
                    encoder_self_att,  # instead of multi-head attention in transformer
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

    def forward(self, X, attn_mask=None):
        enc_out, attns = self.encoder(X, attn_mask)
        return enc_out, attns


class FEDformerDecoder(nn.Module):
    def __init__(
        self,
        n_steps,
        n_pred_steps,
        n_layers,
        n_heads,
        d_model,
        d_ffn,
        d_output,
        moving_avg_window_size,
        dropout,
        version="Fourier",
        modes=32,
        mode_select="random",
        activation="relu",
    ):
        super().__init__()

        if version == "Wavelets":
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=n_steps // 2 + n_pred_steps,
                seq_len_kv=n_steps,
                modes=modes,
                ich=d_model,
                base="legendre",
                activation="tanh",
            )
        elif version == "Fourier":
            decoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=n_steps // 2 + n_pred_steps,
                modes=modes,
                mode_select_method=mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=n_steps // 2 + n_pred_steps,
                seq_len_kv=n_steps,
                modes=modes,
                mode_select_method=mode_select,
                num_heads=n_heads,
            )
        else:
            raise ValueError(f"Unsupported version: {version}. Please choose from ['Wavelets', 'Fourier'].")

        self.decoder = InformerDecoder(
            [
                AutoformerDecoderLayer(
                    decoder_self_att,
                    decoder_cross_att,
                    d_model,
                    n_heads,
                    d_output,
                    d_ffn,
                    moving_avg=moving_avg_window_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_layers)
            ],
            norm_layer=SeasonalLayerNorm(d_model),
            projection=nn.Linear(d_model, d_output, bias=True),
        )

    def forward(self, X, attn_mask=None):
        dec_out, attns = self.decoder(X, attn_mask)
        return dec_out, attns
