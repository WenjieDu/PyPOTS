"""Implementation of pypots.nn.modules.fedformer.autoencoder."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from torch import nn

from ....nn.modules.autoformer import (
    AutoformerDecoderLayer,
    AutoformerEncoderLayer,
    SeasonalLayerNorm,
)
from ....nn.modules.informer import InformerDecoder, InformerEncoder
from .layers import (
    FourierBlock,
    FourierCrossAttention,
    MultiWaveletCross,
    MultiWaveletTransform,
)


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

        self.d_output = d_output
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

    def forward(self, X, memory, trend=None, attn_mask=None, cross_mask=None):
        """Decode X against the encoder output.

        Parameters
        ----------
        X :
            Decoder input of shape [batch_size, n_steps // 2 + n_pred_steps, d_model].

        memory :
            Encoder output of shape [batch_size, n_steps, d_model], read by cross-attention.

        trend :
            Initial trend of shape [batch_size, n_steps // 2 + n_pred_steps, d_output]. Zeros if not given.

        attn_mask, cross_mask :
            Optional masks for self-attention and cross-attention.

        Returns
        -------
        seasonal, trend :
            The seasonal and accumulated trend parts, each [batch_size, n_steps // 2 + n_pred_steps, d_output]; the
            forecast is their sum.
        """
        if trend is None:
            trend = X.new_zeros(*X.shape[:2], self.d_output)
        return self.decoder(X, memory, x_mask=attn_mask, cross_mask=cross_mask, trend=trend)
