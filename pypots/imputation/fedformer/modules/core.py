"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .submodules import MultiWaveletTransform, FourierBlock
from ...autoformer.modules.submodules import (
    AutoformerEncoderLayer,
    AutoCorrelationLayer,
    SeasonalLayerNorm,
)
from ...informer.modules.submodules import InformerEncoder
from ....nn.modules.transformer.embedding import DataEmbedding
from ....utils.metrics import calc_mse


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

        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.enc_embedding = DataEmbedding(
            n_features * 2,
            d_model,
            dropout=dropout,
        )

        if version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
        elif version == "Fourier":
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=n_steps,
                modes=modes,
                mode_select_method=mode_select,
            )
        else:
            raise ValueError(
                f"Unsupported version: {version}. Please choose from ['Wavelets', 'Fourier']."
            )

        self.encoder = InformerEncoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # WDU: the original FEDformer paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I add the
        # embedding layers to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.

        # the same as SAITS, concatenate the time series data and the missing mask for embedding
        input_X = torch.cat([X, masks], dim=2)
        enc_out = self.enc_embedding(input_X)

        # FEDformer encoder processing
        enc_out, attns = self.encoder(enc_out)
        output = self.output_projection(enc_out)

        imputed_data = masks * X + (1 - masks) * output
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # apply SAITS loss function to FEDformer on the imputation task
            ORT_loss = calc_mse(output, X, masks)
            MIT_loss = calc_mse(output, inputs["X_ori"], inputs["indicating_mask"])
            # `loss` is always the item for backward propagating to update the model
            loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
            results["loss"] = loss

        return results
