"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.fft
import torch.nn as nn

from .layers import ResBlock


class TiDE(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_hidden,
        d_feature_encode,
        d_temporal_decoder_hidden,
        dropout,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        n_output_steps = n_steps
        n_output_features = n_features
        d_flatten = n_steps * n_features + n_output_steps * d_feature_encode

        self.feature_encoder = ResBlock(
            n_features,
            d_hidden,
            d_feature_encode,
            dropout,
        )
        self.encoder = TideEncoder(
            n_steps,
            n_features,
            n_layers,
            d_flatten,
            d_hidden,
            dropout,
        )
        self.decoder = TideDecoder(
            n_steps,
            n_steps,
            n_output_features,
            n_layers,
            d_hidden,
            d_feature_encode,
            dropout,
        )
        self.temporal_decoder = ResBlock(
            n_output_features + d_feature_encode,
            d_temporal_decoder_hidden,
            n_output_features,
            dropout,
        )
        self.residual_proj = nn.Linear(n_features, n_output_features)

    def forward(self, X, dynamic):
        bz = X.shape[0]
        feature = self.feature_encoder(dynamic)

        enc_in = torch.cat([X.reshape(bz, -1), feature.reshape(bz, -1)], dim=-1)
        hidden = self.encoder(enc_in)
        decoded = self.decoder(hidden).reshape(hidden.shape[0], self.n_steps, self.n_features)
        temporal_decoder_input = torch.cat([feature, decoded], dim=-1)
        prediction = self.temporal_decoder(temporal_decoder_input)
        prediction += self.residual_proj(X)
        return prediction


class TideEncoder(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_flatten: int,
        d_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.res_hidden = d_hidden
        self.dropout = dropout

        self.encoder_layers = nn.Sequential(
            ResBlock(d_flatten, self.res_hidden, self.d_hidden, dropout),
            *([ResBlock(self.d_hidden, self.res_hidden, self.d_hidden, dropout)] * (self.n_layers - 1)),
        )

    def forward(self, X):
        hidden = self.encoder_layers(X)
        return hidden


class TideDecoder(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_pred_steps: int,
        n_pred_features: int,
        n_layers: int,
        d_hidden: int,
        d_feature_encode,
        dropout: float,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.d_hidden = d_hidden
        res_hidden = d_hidden

        self.decoder_layers = nn.Sequential(
            *([ResBlock(d_hidden, res_hidden, d_hidden, dropout)] * (n_layers - 1)),
            ResBlock(
                d_hidden,
                res_hidden,
                n_pred_features * n_pred_steps,
                dropout,
            ),
        )
        self.final_temporal_decoder = ResBlock(
            n_pred_features + d_feature_encode,
            d_hidden,
            1,
            dropout,
        )
        self.residual_proj = nn.Linear(self.n_steps, self.n_steps)

    def forward(
        self,
        X,
    ):
        dec_out = self.decoder_layers(X).reshape(X.shape[0], self.n_pred_steps, self.n_pred_features)
        return dec_out
