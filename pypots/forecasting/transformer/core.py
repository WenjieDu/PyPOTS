"""
The core wrapper assembles the submodules of Transformer forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.loss import Criterion, MSE
from ...nn.modules.saits import SaitsEmbedding
from ...nn.modules.transformer import TransformerEncoder, TransformerDecoder


class _Transformer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        training_loss: Criterion = MSE(),
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.training_loss = training_loss

        self.encoder_saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )
        self.decoder_saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            n_max_steps=n_pred_steps,
            dropout=dropout,
        )

        self.encoder = TransformerEncoder(
            n_encoder_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )
        self.decoder = TransformerDecoder(
            n_decoder_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )
        self.output_projection = nn.Linear(d_model, n_pred_features)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.training:
            X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
        else:
            batch_size = X.shape[0]
            device = X.device
            X_pred, X_pred_missing_mask = (
                torch.zeros(batch_size, self.n_pred_steps, self.n_pred_features, device=device),
                torch.ones(batch_size, self.n_pred_steps, self.n_pred_features, device=device),
            )

        # apply the SAITS embedding strategy, concatenate X and missing mask for input
        input_X = self.encoder_saits_embedding(X, missing_mask)
        # Transformer encoder processing
        enc_output, _ = self.encoder(input_X)
        input_X_pred = self.decoder_saits_embedding(X_pred, X_pred_missing_mask)
        # Transformer decoder processing
        dec_output, _, _ = self.decoder(input_X_pred, enc_output)
        # project the representation from the d_model-dimensional space to the original data space for output
        forecasting_result = self.output_projection(dec_output)

        # ensemble the results as a dictionary for return
        results = {
            "forecasting_data": forecasting_result,
        }

        # if in training mode, return results with losses
        if self.training:
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = self.training_loss(X_pred, forecasting_result, X_pred_missing_mask)

        return results
