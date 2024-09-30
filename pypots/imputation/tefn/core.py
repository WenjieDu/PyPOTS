"""

"""

# Created by Tianxiang Zhan <zhantianxianguestc@hotmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import calc_mse
from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.modules.tefn import BackboneTEFN


class _TEFN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_fod,
        apply_nonstationary_norm,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_fod = n_fod
        self.apply_nonstationary_norm = apply_nonstationary_norm

        self.model = BackboneTEFN(
            n_steps,
            n_features,
            n_fod,
        )

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # TEFN processing
        out = self.model(X)

        if self.apply_nonstationary_norm:
            # De-Normalization from Non-stationary Transformer
            out = nonstationary_denorm(out, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * out
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            # `loss` is always the item for backward propagating to update the model
            loss = calc_mse(out, inputs["X_ori"], inputs["indicating_mask"])
            results["loss"] = loss

        return results
