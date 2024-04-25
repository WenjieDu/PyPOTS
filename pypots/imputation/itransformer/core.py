"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.saits import SaitsLoss
from ...nn.modules.transformer import TransformerEncoder


class _iTransformer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
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
        self.n_features = n_features
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.embedding = nn.Linear(n_steps, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            n_layers,
            d_model,
            d_ffn,
            n_heads,
            d_k,
            d_v,
            dropout,
            attn_dropout,
        )
        self.output_projection = nn.Linear(d_model, n_steps)

        # apply SAITS loss function to Transformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # apply the SAITS embedding strategy, concatenate X and missing mask for input
        input_X = torch.cat([X.permute(0, 2, 1), missing_mask.permute(0, 2, 1)], dim=1)

        # Transformer encoder processing
        input_X = self.embedding(input_X)
        input_X = self.dropout(input_X)
        enc_output, _ = self.encoder(input_X)
        # project the representation from the d_model-dimensional space to the original data space for output
        reconstruction = self.output_projection(enc_output)
        reconstruction = reconstruction.permute(0, 2, 1)[:, :, : self.n_features]

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
