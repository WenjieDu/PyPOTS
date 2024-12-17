"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.trmf import BackboneTRMF


class _TRMF(nn.Module):
    def __init__(
        self,
        lags,
        K,
        lambda_f,
        lambda_x,
        lambda_w,
        alpha,
        eta,
        max_iter,
        F_step=0.0001,
        X_step=0.0001,
        W_step=0.0001,
    ):
        super().__init__()

        self.backbone = BackboneTRMF(
            lags,
            K,
            lambda_f,
            lambda_x,
            lambda_w,
            alpha,
            eta,
            max_iter,
            F_step,
            X_step,
            W_step,
        )

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        self.backbone.forward(X, missing_mask)
        results = {"loss": 0, "imputed_data": self.backbone.impute_missingness()}
        return results
