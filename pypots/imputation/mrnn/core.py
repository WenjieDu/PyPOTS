"""
PyTorch MRNN model for the time-series imputation task.
This implementation is inspired by the official one https://github.com/jsyoon0823/MRNN.
Some part of the code is from https://github.com/WenjieDu/SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch.nn as nn

from ...nn.modules.mrnn import BackboneMRNN
from ...utils.metrics import calc_rmse


class _MRNN(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size):
        super().__init__()
        self.backbone = BackboneMRNN(n_steps, n_features, rnn_hidden_size)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        RNN_estimation, RNN_imputed_data, FCN_estimation = self.backbone(inputs)

        imputed_data = M * X + (1 - M) * FCN_estimation
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            RNN_loss = calc_rmse(RNN_estimation, X, M)
            FCN_loss = calc_rmse(FCN_estimation, RNN_imputed_data)
            reconstruction_loss = RNN_loss + FCN_loss
            results["loss"] = reconstruction_loss

        return results
