"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple

import torch
import torch.nn as nn

from .layers import MrnnFcnRegression


class BackboneMRNN(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.f_rnn = nn.GRU(3, self.rnn_hidden_size, batch_first=True)
        self.b_rnn = nn.GRU(3, self.rnn_hidden_size, batch_first=True)
        self.concated_hidden_project = nn.Linear(self.rnn_hidden_size * 2, 1)
        self.fcn_regression = MrnnFcnRegression(n_features)

    def gene_hidden_states(self, inputs, feature_idx):
        X_f = inputs["forward"]["X"][:, :, feature_idx].unsqueeze(dim=2)
        M_f = inputs["forward"]["missing_mask"][:, :, feature_idx].unsqueeze(dim=2)
        D_f = inputs["forward"]["deltas"][:, :, feature_idx].unsqueeze(dim=2)
        X_b = inputs["backward"]["X"][:, :, feature_idx].unsqueeze(dim=2)
        M_b = inputs["backward"]["missing_mask"][:, :, feature_idx].unsqueeze(dim=2)
        D_b = inputs["backward"]["deltas"][:, :, feature_idx].unsqueeze(dim=2)
        device = X_f.device
        batch_size = X_f.size()[0]

        f_hidden_state_0 = torch.zeros((1, batch_size, self.rnn_hidden_size), device=device)
        b_hidden_state_0 = torch.zeros((1, batch_size, self.rnn_hidden_size), device=device)
        f_input = torch.cat([X_f, M_f, D_f], dim=2)
        b_input = torch.cat([X_b, M_b, D_b], dim=2)
        hidden_states_f, _ = self.f_rnn(f_input, f_hidden_state_0)
        hidden_states_b, _ = self.b_rnn(b_input, b_hidden_state_0)
        hidden_states_b = torch.flip(hidden_states_b, dims=[1])

        feature_estimation = self.concated_hidden_project(torch.cat([hidden_states_f, hidden_states_b], dim=2))

        return feature_estimation, hidden_states_f, hidden_states_b

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = inputs["forward"]["X"]
        M = inputs["forward"]["missing_mask"]

        feature_collector = []
        for f in range(self.n_features):
            feat_estimation, hid_states_f, hid_states_b = self.gene_hidden_states(inputs, f)
            feature_collector.append(feat_estimation)

        RNN_estimation = torch.concat(feature_collector, dim=2)
        RNN_imputed_data = M * X + (1 - M) * RNN_estimation
        FCN_estimation = self.fcn_regression(X, M, RNN_imputed_data)
        return RNN_estimation, RNN_imputed_data, FCN_estimation
