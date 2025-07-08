"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from .layers import TimesBlock


class BackboneTimesNet(nn.Module):
    def __init__(
        self,
        n_layers,
        n_steps,
        n_pred_steps,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_layers = n_layers

        self.n_pred_steps = n_pred_steps
        self.model = nn.ModuleList(
            [TimesBlock(n_steps, n_pred_steps, top_k, d_model, d_ffn, n_kernels) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X) -> torch.Tensor:

        enc_out = X
        for i in range(self.n_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        return enc_out
