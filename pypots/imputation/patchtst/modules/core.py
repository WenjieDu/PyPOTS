"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .submodules import PatchEmbedding, FlattenHead
from ....nn.modules.transformer.attention import ScaledDotProductAttention
from ....nn.modules.transformer.auto_encoder import EncoderLayer
from ....utils.metrics import calc_mse


class _PatchTST(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ffn: int,
        d_k: int,
        d_v: int,
        patch_len: int,
        stride: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        patch_num = int((n_steps - patch_len) / stride + 2)
        head_nf = d_model * patch_num
        padding = stride

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = nn.Linear(n_features * 2, d_model)
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        self.encoder = nn.ModuleList(
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
        self.head = FlattenHead(head_nf, n_steps, dropout)
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # WDU: the original PatchTST paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I add the
        # embedding layers to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project the seasonal and trend from the hidden space to the original space.

        # do patching and embedding
        input_X = self.embedding(torch.cat([X, masks], dim=2))
        enc_out = self.patch_embedding(input_X.permute(0, 2, 1))

        # PatchTST encoder processing
        # z: [bs * d_model x patch_num x d_model]
        for i in range(self.n_layers):
            enc_out, _ = self.encoder[i](enc_out)
        # z: [bs x d_model x patch_num x d_model]
        enc_out = enc_out.reshape(
            -1, self.d_model, enc_out.shape[-2], enc_out.shape[-1]
        )
        # z: [bs x d_model x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # project back the original data space
        dec_out = self.head(enc_out)  # z: [bs x d_model x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.output_projection(dec_out)

        imputed_data = masks * X + (1 - masks) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
