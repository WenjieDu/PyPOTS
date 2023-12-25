"""
PyTorch MRNN model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from .submodules import FCN_Regression
from ....utils.metrics import calc_rmse


class _MRNN(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, device):
        super().__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.f_rnn = nn.GRUCell(3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(3, self.rnn_hidden_size)
        self.concated_hidden_project = nn.Linear(self.rnn_hidden_size * 2, 1)
        self.fcn_regression = FCN_Regression(feature_num)

    def gene_hidden_states(self, inputs, feature_idx):
        X_f = inputs["forward"]["X"][:, :, feature_idx].unsqueeze(dim=2)
        M_f = inputs["forward"]["missing_mask"][:, :, feature_idx].unsqueeze(dim=2)
        D_f = inputs["forward"]["deltas"][:, :, feature_idx].unsqueeze(dim=2)
        X_b = inputs["backward"]["X"][:, :, feature_idx].unsqueeze(dim=2)
        M_b = inputs["backward"]["missing_mask"][:, :, feature_idx].unsqueeze(dim=2)
        D_b = inputs["backward"]["deltas"][:, :, feature_idx].unsqueeze(dim=2)
        device = X_f.device

        hidden_state_f = torch.zeros(
            (X_f.size()[0], self.rnn_hidden_size), device=device
        )
        hidden_state_b = torch.zeros(
            (X_f.size()[0], self.rnn_hidden_size), device=device
        )

        hidden_states_f_collector = []
        hidden_states_b_collector = []
        for t in range(self.seq_len):
            x_f, m_f, d_f = X_f[:, t, :], M_f[:, t, :], D_f[:, t, :]
            input_f = torch.cat([x_f, m_f, d_f], dim=1)
            x_b, m_b, d_b = X_b[:, t, :], M_b[:, t, :], D_b[:, t, :]
            input_b = torch.cat([x_b, m_b, d_b], dim=1)
            hidden_state_f = self.f_rnn(input_f, hidden_state_f)
            hidden_state_b = self.b_rnn(input_b, hidden_state_b)

            hidden_states_f_collector.append(hidden_state_f)
            hidden_states_b_collector.append(hidden_state_b)

        hidden_states_f = torch.stack(hidden_states_f_collector, dim=1)
        hidden_states_b = torch.stack(hidden_states_b_collector, dim=1)
        hidden_states_b = torch.flip(hidden_states_b, dims=[1])

        feature_estimation = self.concated_hidden_project(
            torch.cat([hidden_states_f, hidden_states_b], dim=2)
        )

        return feature_estimation, hidden_states_f, hidden_states_b

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        feature_collector = []
        for f in range(self.feature_num):
            feat_estimation, hid_states_f, hid_states_b = self.gene_hidden_states(
                inputs, f
            )
            feature_collector.append(feat_estimation)
        RNN_estimation = torch.concat(feature_collector, dim=2)
        RNN_imputed_data = M * X + (1 - M) * RNN_estimation
        FCN_estimation = self.fcn_regression(X, M, RNN_imputed_data)

        imputed_data = M * X + (1 - M) * FCN_estimation
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            reconstruction_loss = calc_rmse(FCN_estimation, X, M) + calc_rmse(
                RNN_estimation, X, M
            )
            results["loss"] = reconstruction_loss

        return results
