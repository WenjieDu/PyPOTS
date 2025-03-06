"""

"""

# Created by Tianxiang Zhan <zhantianxianguestc@hotmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .layers import EvidenceMachineKernel


class BackboneTEFN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_fod,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_fod = n_fod

        self.T_model = EvidenceMachineKernel(self.n_steps + self.n_pred_steps, self.n_fod)
        self.C_model = EvidenceMachineKernel(self.n_features, self.n_fod)

    def forward(self, X) -> torch.Tensor:
        X = self.T_model(X.permute(0, 2, 1)).permute(0, 2, 1, 3) + self.C_model(X)
        X = torch.einsum("btcf->btc", X)
        return X
