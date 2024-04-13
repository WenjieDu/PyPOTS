"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules.fedformer import FEDformerEncoder
from ...nn.modules.saits import SaitsLoss
from ...nn.modules.transformer.embedding import DataEmbedding


class _FEDformer(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        n_heads,
        d_model,
        d_ffn,
        moving_avg_window_size,
        dropout,
        version="Fourier",
        modes=32,
        mode_select="random",
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        activation="relu",
    ):
        super().__init__()

        self.enc_embedding = DataEmbedding(
            n_features * 2,
            d_model,
            dropout=dropout,
        )

        self.encoder = FEDformerEncoder(
            n_steps,
            n_layers,
            n_heads,
            d_model,
            d_ffn,
            moving_avg_window_size,
            dropout,
            version,
            modes,
            mode_select,
            activation,
        )
        self.output_projection = nn.Linear(d_model, n_features)

        # apply SAITS loss function to ETSformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original FEDformer paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I add the
        # embedding layers to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.

        # the same as SAITS, concatenate the time series data and the missing mask for embedding
        input_X = torch.cat([X, missing_mask], dim=2)
        enc_out = self.enc_embedding(input_X)

        # FEDformer encoder processing
        enc_out, attns = self.encoder(enc_out)
        reconstruction = self.output_projection(enc_out)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
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
