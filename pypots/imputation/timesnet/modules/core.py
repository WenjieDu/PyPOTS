"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .embedding import DataEmbedding
from .layer import TimesBlock
from ....nn.functional import nonstationary_norm, nonstationary_denorm
from ....utils.metrics import calc_mse


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

        self.pred_len = 0  # for the imputation task, the pred_len is always 0
        self.model = nn.ModuleList(
            [
                TimesBlock(n_steps, self.pred_len, top_k, d_model, d_ffn, n_kernels)
                for _ in range(n_layers)
            ]
        )
        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # for the imputation task, the output dim is the same as input dim
        c_out = n_features
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, masks)

        # embedding
        enc_out = self.enc_embedding(X)  # [B,T,C]
        # TimesNet
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back the original data space
        dec_out = self.projection(enc_out)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = nonstationary_denorm(dec_out, means, stdev)

        imputed_data = masks * X + (1 - masks) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
