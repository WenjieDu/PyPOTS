"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.functional import calc_mse
from ...nn.modules.timesnet import BackboneTimesNet
from ...nn.modules.transformer.embedding import DataEmbedding


class _TimesNet(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_features,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
        dropout,
        apply_nonstationary_norm,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers
        self.apply_nonstationary_norm = apply_nonstationary_norm

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
            n_max_steps=n_steps,
        )
        self.model = BackboneTimesNet(
            n_layers,
            n_steps,
            0,  # n_pred_steps should be 0 for the imputation task
            top_k,
            d_model,
            d_ffn,
            n_kernels,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # embedding
        input_X = self.enc_embedding(X)  # [B,T,C]
        # TimesNet processing
        enc_out = self.model(input_X)

        # project back the original data space
        dec_out = self.projection(enc_out)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = nonstationary_denorm(dec_out, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if self.training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
