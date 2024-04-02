"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .submodules import (
    SeasonalLayerNorm,
    AutoformerEncoderLayer,
    AutoCorrelation,
    AutoCorrelationLayer,
)
from ...informer.modules.submodules import InformerEncoder
from ....nn.modules.transformer.embedding import DataEmbedding
from ....utils.metrics import calc_mse


class _Autoformer(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        n_heads,
        d_model,
        d_ffn,
        factor,
        moving_avg_window_size,
        dropout,
        activation="relu",
        output_attention=False,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers
        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
            with_pos=False,
        )
        self.encoder = InformerEncoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, dropout, output_attention),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ffn,
                    moving_avg_window_size,
                    dropout,
                    activation,
                )
                for _ in range(n_layers)
            ],
            norm_layer=SeasonalLayerNorm(d_model),
        )

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # embedding
        enc_out = self.enc_embedding(X)  # [B,T,C]

        # Autoformer encoder processing
        enc_out, attns = self.encoder(enc_out)

        # project back the original data space
        dec_out = self.projection(enc_out)

        imputed_data = masks * X + (1 - masks) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
