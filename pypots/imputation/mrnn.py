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
from torch.utils.data import DataLoader

from pypots.data.base import BaseDataset
from pypots.data.dataset_for_brits import DatasetForBRITS
from pypots.data.integration import mcar, masked_fill
from pypots.imputation.base import BaseNNImputer
from pypots.imputation.brits import FeatureRegression
from pypots.utils.metrics import cal_mae
from pypots.utils.metrics import cal_rmse


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
            )  # FCN estimation is output estimation
            reconstruction_loss += cal_rmse(FCN_estimation, x, m) + cal_rmse(
                RNN_estimation, x, m
            )
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [estimations, reconstruction_loss]

    def forward(self, inputs):
        imputed_data, [_, reconstruction_loss] = self.impute(inputs)
        reconstruction_loss /= self.seq_len

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
        imputation_MAE = cal_mae(
            imputed_data, inputs["X_holdout"], inputs["indicating_mask"]
        )

        ret_dict = {
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "imputed_data": imputed_data,
        }
        return ret_dict


class MRNN(BaseNNImputer):
    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        learning_rate=1e-3,
        epochs=100,
        patience=10,
        batch_size=32,
        weight_decay=1e-5,
        device=None,
    ):
        super().__init__(
            learning_rate, epochs, patience, batch_size, weight_decay, device
        )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.rnn_hidden_size = rnn_hidden_size

        self.model = _MRNN(
            self.n_steps, self.n_features, self.rnn_hidden_size, self.device
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_X, val_X=None):
        train_X = self.check_input(self.n_steps, self.n_features, train_X)
        if val_X is not None:
            val_X = self.check_input(self.n_steps, self.n_features, val_X)

        training_set = DatasetForBRITS(train_X)
        training_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True
        )
        if val_X is None:
            self._train_model(training_loader)
        else:
            val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(
                val_X, 0.2
            )
            val_X = masked_fill(val_X, 1 - val_X_missing_mask, torch.nan)
            val_set = DatasetForBRITS(val_X)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(
                training_loader, val_loader, val_X_intact, val_X_indicating_mask
            )

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

    def assemble_input_data(self, data):
        """Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        indices, X_intact, X, missing_mask, indicating_mask = data

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def impute(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = {"X": data[1], "missing_mask": data[2]}
                imputed_data, _ = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
