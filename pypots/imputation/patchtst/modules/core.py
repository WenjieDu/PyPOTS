"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from .submodules import PatchEmbedding, FlattenHead
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
        head_nf: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers

        padding = stride
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        # self.encoder = Encoder(
        #     n_layers,
        #     n_steps,
        #     n_features,
        #     d_model,
        #     d_ffn,
        #     n_heads,
        #     d_k,
        #     d_v,
        #     dropout,
        #     attn_dropout,
        # )
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_ffn,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = FlattenHead(n_features, head_nf, n_steps, dropout)

        # for the imputation task, the output dim is the same as input dim
        self.projection = nn.Linear(d_model, n_features)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]

        # do patching and embedding
        x_enc = X.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # PatchTST encoder processing
        # z: [bs * nvars x patch_num x d_model]
        for i in range(self.n_layers):
            enc_out, _ = self.encoder[i](enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # project back the original data space
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        imputed_data = masks * X + (1 - masks) * dec_out
        results = {
            "imputed_data": imputed_data,
        }

        if training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(dec_out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
