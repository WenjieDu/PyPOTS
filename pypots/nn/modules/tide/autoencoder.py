"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.fft
import torch.nn as nn

from .layers import ResBlock


class TideEncoder(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_hidden: int,
        d_feature_encode: int,
        dropout: float,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.res_hidden = d_hidden
        self.dropout = dropout
        self.d_feature_encode = d_feature_encode

        flatten_dim = (
            self.n_steps + (self.n_steps + self.n_pred_steps) * self.d_feature_encode
        )
        self.feature_encoder = ResBlock(
            self.n_features, self.res_hidden, self.d_feature_encode, dropout
        )
        self.encoder_layers = nn.Sequential(
            ResBlock(flatten_dim, self.res_hidden, self.d_hidden, dropout),
            *(
                [ResBlock(self.d_hidden, self.res_hidden, self.d_hidden, dropout)]
                * (self.n_layers - 1)
            )
        )

    def forward(self, X, dynamic):
        feature = self.feature_encoder(dynamic)
        hidden = self.encoder_layers(
            torch.cat([X, feature.reshape(feature.shape[0], -1)], dim=-1)
        )
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
        self.d_hidden = d_hidden
        res_hidden = d_hidden

        self.decoder_layers = nn.Sequential(
            *([ResBlock(d_hidden, res_hidden, d_hidden, dropout)] * (n_layers - 1)),
            ResBlock(
                d_hidden,
                res_hidden,
                n_pred_features * n_pred_steps,
                dropout,
            )
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
        feature_encoding,
        hidden_stats,
    ):
        decoded = self.decoder_layers(hidden_stats).reshape(
            hidden_stats.shape[0], self.n_pred_steps, self.n_pred_features
        )
        dec_out = self.temporalDecoder(
            torch.cat([feature_encoding[:, self.n_steps :], decoded], dim=-1)
        ).squeeze(-1) + self.residual_proj(X)
        return dec_out
