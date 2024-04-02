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

from typing import Tuple

import torch
import torch.nn as nn

from ....nn.modules.transformer import EncoderLayer, PositionalEncoding
from ....nn.modules.transformer.attention import ScaledDotProductAttention
from ....utils.metrics import calc_mae


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_time: int,
        d_feature: int,
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

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_ffn,
                    n_heads,
                    d_k,
                    d_v,
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_positions=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def _process(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        X, masks = inputs["X"], inputs["missing_mask"]
        input_X = torch.cat([X, masks], dim=2)
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.layer_stack:
            enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, learned_presentation = self._process(inputs)

        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            ORT_loss = calc_mae(learned_presentation, X, masks)
            MIT_loss = calc_mae(
                learned_presentation, inputs["X_ori"], inputs["indicating_mask"]
            )
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
            results["loss"] = loss

        return results
