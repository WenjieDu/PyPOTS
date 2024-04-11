"""
The implementation of Transformer for the partially-observed time-series imputation task.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.saits import SaitsLoss
from ...nn.modules.transformer import TransformerEncoder


class _Transformer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        d_feature: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.encoder = TransformerEncoder(
            n_steps,
            actual_d_feature,
            n_layers,
            d_model,
            d_ffn,
            n_heads,
            d_k,
            d_v,
            dropout,
            attn_dropout,
        )
        self.output_projection = nn.Linear(d_model, d_feature)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # apply the SAITS embedding strategy, concatenate X and missing mask for input
        input_X = torch.cat([X, missing_mask], dim=2)

        # Transformer encoder processing
        enc_output = self.encoder(input_X)
        # project the representation from the d_model-dimensional space to the original data space for output
        reconstruction = self.output_projection(enc_output)

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        # ensemble the results as a dictionary for return
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            loss, ORT_loss, MIT_loss = self.saits_loss_func(
                reconstruction, X_ori, missing_mask, indicating_mask
            )
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

        return results
