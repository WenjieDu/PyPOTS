"""
The core wrapper assembles the submodules of ETSformer imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.etsformer import (
    ETSformerEncoderLayer,
    ETSformerEncoder,
    ETSformerDecoderLayer,
    ETSformerDecoder,
)
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _ETSformer(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_encoder_layers,
        n_decoder_layers,
        d_model,
        n_heads,
        d_ffn,
        dropout,
        top_k,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        activation="sigmoid",
    ):
        super().__init__()

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
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
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_encoder_layers)
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
                for _ in range(n_decoder_layers)
            ],
        )

        # apply SAITS loss function to ETSformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original ETSformer paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        res = self.saits_embedding(X, missing_mask)

        # ETSformer encoder processing
        level, growths, seasons = self.encoder(res, X, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        reconstruction = level[:, -1:] + growth + season

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

        return results
