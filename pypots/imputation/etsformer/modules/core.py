"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .submodules import (
    ETSformerEncoderLayer,
    ETSformerEncoder,
    ETSformerDecoderLayer,
    ETSformerDecoder,
)
from ....nn.modules.transformer.embedding import DataEmbedding
from ....utils.metrics import calc_mse


class _ETSformer(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_e_layers,
        n_d_layers,
        n_heads,
        d_model,
        d_ffn,
        dropout,
        top_k,
        activation="sigmoid",
    ):
        super().__init__()

        self.n_steps = n_steps

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
        )

        # Encoder
        self.encoder = ETSformerEncoder(
            [
                ETSformerEncoderLayer(
                    d_model,
                    n_heads,
                    n_features,
                    n_steps,
                    n_steps,
                    top_k,
                    dim_feedforward=d_ffn,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_e_layers)
            ]
        )
        # Decoder
        self.decoder = ETSformerDecoder(
            [
                ETSformerDecoderLayer(
                    d_model,
                    n_heads,
                    n_features,
                    n_steps,
                    dropout=dropout,
                )
                for _ in range(n_d_layers)
            ],
        )
        # self.transform = Transform(sigma=0.2)  # for forecasting

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # embedding
        res = self.enc_embedding(X)

        # ETSformer encoder processing
        level, growths, seasons = self.encoder(res, X, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        output = level[:, -1:] + growth + season

        imputed_data = masks * X + (1 - masks) * output
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(output, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
