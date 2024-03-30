"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from math import ceil

import torch
import torch.nn as nn
from einops import rearrange

from .submodules import CrossformerEncoder, ScaleBlock
from ...patchtst.modules.submodules import FlattenHead, PatchEmbedding
from ....utils.metrics import calc_mse


class _Crossformer(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        n_heads,
        d_model,
        d_ffn,
        factor,
        seg_len,
        win_size,
        dropout,
    ):
        super().__init__()

        self.n_features = n_features

        # The padding operation to handle invisible sgemnet length
        pad_in_len = ceil(1.0 * n_steps / seg_len) * seg_len
        in_seg_num = pad_in_len // seg_len
        out_seg_num = ceil(in_seg_num / (win_size ** (n_layers - 1)))
        head_nf = d_model * out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            d_model,
            seg_len,
            seg_len,
            pad_in_len - n_steps,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, n_features, in_seg_num, d_model)
        )
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

        self.head = FlattenHead(head_nf, n_steps, dropout)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # embedding
        x_enc = self.enc_value_embedding(X.permute(0, 2, 1))

        # Crossformer processing
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=self.n_features
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # project back the original data space
        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        imputed_data = masks * X + (1 - masks) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
