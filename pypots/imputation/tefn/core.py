"""

"""

# Created by Tianxiang Zhan <zhantianxianguestc@hotmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional import calc_mse
from ...nn.functional import nonstationary_norm, nonstationary_denorm
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.tefn import BackboneTEFN


class _TEFN(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_fod: int,
        apply_nonstationary_norm: bool = False,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.seq_len = n_steps
        self.n_fod = n_fod
        self.apply_nonstationary_norm = apply_nonstationary_norm

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
        )
        self.model = BackboneTEFN(
            n_steps,
            n_features,
            0,
            n_fod,
        )

        # apply SAITS loss function to Transformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.apply_nonstationary_norm:
            # Normalization from Non-stationary Transformer
            X, means, stdev = nonstationary_norm(X, missing_mask)

        # WDU: the original FITS paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)

        # TEFN processing
        out = self.model(enc_out)

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
