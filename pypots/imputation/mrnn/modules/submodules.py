"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ...brits.modules import FeatureRegression


class FCN_Regression(nn.Module):
    def __init__(self, feature_num, rnn_hid_size):
        super().__init__()
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
