"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.fft
import torch.nn as nn

from .embedding import DataEmbedding
from .layer import TimesBlock
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
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers

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

        # Normalization from Non-stationary Transformer
        means = torch.sum(X, dim=1) / torch.sum(masks == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = X - means
        x_enc = x_enc.masked_fill(masks == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(masks == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back the original data space
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )

        imputed_data = masks * X + (1 - masks) * dec_out

        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_intact"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
