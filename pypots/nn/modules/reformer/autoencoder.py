"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .layers import ReformerLayer


class ReformerEncoder(nn.Module):
    def __init__(
        self,
        n_steps,
        n_layers,
        d_model,
        n_heads,
        bucket_size,
        n_hashes,
        causal,
        d_ffn,
        dropout,
    ):
        super().__init__()

        assert (
            n_steps % (bucket_size * 2) == 0
        ), f"Sequence length ({n_steps}) needs to be divisible by target bucket size {bucket_size} x 2"

        self.enc_layer_stack = nn.ModuleList(
            [
                ReformerLayer(
                    d_model,
                    n_heads,
                    bucket_size,
                    n_hashes,
                    causal,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        enc_output = x

        for layer in self.enc_layer_stack:
            enc_output = layer(enc_output)

        return enc_output
