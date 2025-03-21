"""
The core wrapper assembles the submodules of M-RNN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union

import torch.nn as nn

from ...nn.modules.loss import Criterion, RMSE
from ...nn.modules.mrnn import BackboneMRNN


class _MRNN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        training_loss: Union[Criterion, type] = RMSE(),
    ):
        super().__init__()
        self.backbone = BackboneMRNN(n_steps, n_features, rnn_hidden_size)
        self.training_loss = training_loss

    def forward(self, inputs: dict) -> dict:
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        RNN_estimation, RNN_imputed_data, FCN_estimation = self.backbone(inputs)

        imputed_data = M * X + (1 - M) * FCN_estimation
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            RNN_loss = self.training_loss(RNN_estimation, X, M)
            FCN_loss = self.training_loss(FCN_estimation, RNN_imputed_data)
            reconstruction_loss = RNN_loss + FCN_loss
            results["loss"] = reconstruction_loss

        return results
