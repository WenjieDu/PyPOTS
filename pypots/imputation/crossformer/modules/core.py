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
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

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
            torch.randn(1, d_model, in_seg_num, d_model)
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
        self.embedding = nn.Linear(n_features * 2, d_model)
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # WDU: the original Crossformer paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I add the
        # embedding layers to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        # embedding
        input_X = self.embedding(torch.cat([X, masks], dim=2))
        x_enc = self.enc_value_embedding(input_X.permute(0, 2, 1))

        # Crossformer processing
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=self.d_model
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # project back the original data space
        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        output = self.output_projection(dec_out)

        imputed_data = masks * X + (1 - masks) * output
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # apply SAITS loss function to Crossformer on the imputation task
            ORT_loss = calc_mse(output, X, masks)
            MIT_loss = calc_mse(output, inputs["X_ori"], inputs["indicating_mask"])
            # `loss` is always the item for backward propagating to update the model
            loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
            results["loss"] = loss

        return results
