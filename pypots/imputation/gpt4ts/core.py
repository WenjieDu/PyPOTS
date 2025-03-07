"""
The core wrapper assembles the submodules of GPT4TS imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Callable

import torch.nn as nn

from ...nn.functional import calc_mse
from ...nn.modules.gpt4ts import BackboneGPT4TS


class _GPT4TS(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        patch_size: int,
        patch_stride: int,
        train_gpt_mlp: bool,
        d_ffn: int,
        dropout: float,
        embed: str,
        freq: str,
        loss_func: Callable = calc_mse,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.loss_func = loss_func

        self.backbone = BackboneGPT4TS(
            "imputation",
            n_steps,
            n_features,
            0,
            n_features,
            n_layers,
            patch_size,
            patch_stride,
            train_gpt_mlp,
            d_ffn,
            dropout,
            embed,
            freq,
        )

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # GPT4TS backbone processing
        reconstruction = self.backbone(X, mask=missing_mask)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            # `loss` is always the item for backward propagating to update the model
            loss = self.loss_func(reconstruction, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
