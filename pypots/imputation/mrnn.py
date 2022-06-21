"""
PyTorch MRNN model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from pypots.imputation.brits import FeatureRegression
from pypots.utils.metrics import cal_mae, cal_rmse


class FCN_Regression(nn.Module):
    def __init__(self, feature_num, rnn_hid_size):
        super(FCN_Regression, self).__init__()
        self.feat_reg = FeatureRegression(rnn_hid_size * 2)
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))  # bias beta
        self.final_linear = nn.Linear(feature_num, feature_num)

        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x_t, m_t, target):
        h_t = F.tanh(
            F.linear(x_t, self.U * self.m)
            + F.linear(target, self.V1 * self.m)
            + F.linear(m_t, self.V2)
            + self.beta
        )
        x_hat_t = self.final_linear(h_t)
        return x_hat_t


class MRNN(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(MRNN, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = kwargs["device"]

        self.f_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.rnn_cells = {"forward": self.f_rnn, "backward": self.b_rnn}
        self.concated_hidden_project = nn.Linear(
            self.rnn_hidden_size * 2, self.feature_num
        )
        self.fcn_regression = FCN_Regression(feature_num, rnn_hidden_size)

    def gene_hidden_states(self, data, direction):
        values = data[direction]["X"]
        masks = data[direction]["missing_mask"]
        deltas = data[direction]["deltas"]

        hidden_states_collector = []
        hidden_state = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            inputs = torch.cat([x, m, d], dim=1)
            hidden_state = self.rnn_cells[direction](inputs, hidden_state)
            hidden_states_collector.append(hidden_state)
        return hidden_states_collector

    def impute(self, data):
        hidden_states_f = self.gene_hidden_states(data, "forward")
        hidden_states_b = self.gene_hidden_states(data, "backward")[::-1]

        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]

        reconstruction_loss = 0
        estimations = []
        for i in range(
            self.seq_len
        ):  # calculating estimation loss for times can obtain better results than once
            x = values[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(
                x, m, RNN_imputed_data
            )  # FCN estimation is output extimation
            reconstruction_loss += cal_rmse(FCN_estimation, x, m) + cal_rmse(
                RNN_estimation, x, m
            )
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [estimations, reconstruction_loss]

    def forward(self, data, stage):
        values = data["forward"]["X"]
        masks = data["forward"]["missing_mask"]
        imputed_data, [estimations, reconstruction_loss] = self.impute(data)
        reconstruction_loss /= self.seq_len
        reconstruction_MAE = cal_mae(estimations.detach(), values, masks)

        if stage == "val":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = cal_mae(
                imputed_data, data["X_holdout"], data["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        ret_dict = {
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "imputation_MAE": imputation_MAE,
            "imputed_data": imputed_data,
        }
        if "X_holdout" in data:
            ret_dict["X_holdout"] = data["X_holdout"]
            ret_dict["indicating_mask"] = data["indicating_mask"]
        return ret_dict
