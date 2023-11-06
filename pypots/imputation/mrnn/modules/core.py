"""
PyTorch MRNN model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from .submodules import FCN_Regression
from ....utils.metrics import cal_rmse


class _MRNN(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, device):
        super().__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.f_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.concated_hidden_project = nn.Linear(
            self.rnn_hidden_size * 2, self.feature_num
        )
        self.fcn_regression = FCN_Regression(feature_num, rnn_hidden_size)

    def gene_hidden_states(self, inputs, direction):
        X = inputs[direction]["X"]
        masks = inputs[direction]["missing_mask"]
        deltas = inputs[direction]["deltas"]
        device = X.device

        hidden_states_collector = []
        hidden_state = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)

        for t in range(self.seq_len):
            x = X[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            inputs = torch.cat([x, m, d], dim=1)
            if direction == "forward":
                hidden_state = self.f_rnn(inputs, hidden_state)
            else:
                hidden_state = self.b_rnn(inputs, hidden_state)
            hidden_states_collector.append(hidden_state)
        return hidden_states_collector

    def forward(self, inputs, training=True):
        hidden_states_f = self.gene_hidden_states(inputs, "forward")
        hidden_states_b = self.gene_hidden_states(inputs, "backward")[::-1]

        X = inputs["forward"]["X"]
        masks = inputs["forward"]["missing_mask"]

        reconstruction_loss = 0
        estimations = []
        for i in range(
            self.seq_len
        ):  # calculating estimation loss for times can obtain better results than once
            x = X[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(
                x, m, RNN_imputed_data
            )  # FCN estimation is output estimation
            reconstruction_loss += cal_rmse(FCN_estimation, x, m) + cal_rmse(
                RNN_estimation, x, m
            )
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * X + (1 - masks) * estimations

        if not training:
            # if not in training mode, return the classification result only
            return {
                "imputed_data": imputed_data,
            }

        reconstruction_loss /= self.seq_len

        ret_dict = {
            "loss": reconstruction_loss,
            "imputed_data": imputed_data,
        }
        return ret_dict
