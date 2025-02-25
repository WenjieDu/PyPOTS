"""
The core wrapper assembles the submodules of Crossformer imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from math import ceil

import torch
import torch.nn as nn
from einops import rearrange

from ...nn.modules.crossformer import CrossformerEncoder, ScaleBlock
from ...nn.modules.patchtst import PredictionHead, PatchEmbedding
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _Crossformer(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        factor,
        seg_len,
        win_size,
        dropout,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.d_model = d_model

        # The padding operation to handle invisible sgemnet length
        pad_in_len = ceil(1.0 * n_steps / seg_len) * seg_len
        in_seg_num = pad_in_len // seg_len
        out_seg_num = ceil(in_seg_num / (win_size ** (n_layers - 1)))

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            d_model,
            seg_len,
            seg_len,
            pad_in_len - n_steps,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, d_model, in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = CrossformerEncoder(
            [
                ScaleBlock(
                    1 if layer == 0 else win_size,
                    d_model,
                    n_heads,
                    d_ffn,
                    1,
                    dropout,
                    in_seg_num if layer == 0 else ceil(in_seg_num / win_size**layer),
                    factor,
                )
                for layer in range(n_layers)
            ]
        )

        self.head = PredictionHead(d_model, out_seg_num, n_steps, dropout)
        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=False,
        )
        self.output_projection = nn.Linear(d_model, n_features)

        # apply SAITS loss function to Crossformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original Crossformer paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        input_X = self.saits_embedding(X, missing_mask)

        x_enc = self.enc_value_embedding(input_X.permute(0, 2, 1))
        x_enc = rearrange(x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=self.d_model)
        x_enc += self.enc_pos_embedding

        # Crossformer processing
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # project back the original data space
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        reconstruction = self.output_projection(dec_out)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

        return results
