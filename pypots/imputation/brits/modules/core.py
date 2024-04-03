"""
The implementation of BRITS for the partially-observed time-series imputation task.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS. The bugs in the original implementation
are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple, Union

import torch
import torch.nn as nn

from .submodules import FeatureRegression
from ....nn.modules.rnn import TemporalDecay
from ....utils.metrics import calc_mae


class RITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    device :
        specify running the model on which device, CPU/GPU

    rnn_cell :
        the LSTM cell to model temporal data

    temp_decay_h :
        the temporal decay module to decay RNN hidden state

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist_reg :
        the temporal-regression module to project RNN hidden state into the raw feature space

    feat_reg :
        the feature-regression module

    combining_weight :
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    device :
        specify running the model on which device, CPU/GPU

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def impute(
        self, inputs: dict, direction: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The imputation function.
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data :
            [batch size, sequence length, feature number]

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        values = inputs[direction]["X"]  # feature values
        masks = inputs[direction]["missing_mask"]  # missing masks
        deltas = inputs[direction]["deltas"]  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(values.device)

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += calc_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += calc_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += calc_mae(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, estimations, hidden_states, reconstruction_loss

    def forward(self, inputs: dict, direction: str = "forward") -> dict:
        """Forward processing of the NN module.
        Parameters
        ----------
        inputs :
            The input data.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        dict,
            A dictionary includes all results.

        """
        imputed_data, estimations, hidden_state, reconstruction_loss = self.impute(inputs, direction)
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        ret_dict = {
            "consistency_loss": torch.tensor(
                0.0, device=imputed_data.device
            ),  # single direction, has no consistency loss
            "reconstruction_loss": reconstruction_loss,
            "imputed_data": imputed_data,
            "reconstructed_data": estimations,
            "final_hidden_state": hidden_state,
        }
        return ret_dict


class _BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rits_f: RITS object
        the forward RITS model

    rits_b: RITS object
        the backward RITS model

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, device)

    @staticmethod
    def _get_consistency_loss(
        pred_f: torch.Tensor, pred_b: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f :
            The imputation from the forward RITS.

        pred_b :
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.

        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def _reverse(ret: dict) -> dict:
        """Reverse the array values on the time dimension in the given dictionary.

        Parameters
        ----------
        ret :

        Returns
        -------
        dict,
            A dictionary contains values reversed on the time dimension from the given dict.

        """

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def forward(self, inputs: dict, training: bool = True) -> dict:
        # Results from the forward RITS.
        ret_f = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2
        reconstructed_data = (ret_f["reconstructed_data"] + ret_b["reconstructed_data"]) / 2

        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            consistency_loss = self._get_consistency_loss(
                ret_f["imputed_data"], ret_b["imputed_data"]
            )
            results["consistency_loss"] = consistency_loss
            loss = (
                consistency_loss
                + ret_f["reconstruction_loss"]
                + ret_b["reconstruction_loss"]
            )

            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss
            results['reconstructed_data'] = reconstructed_data
            results['f_reconstructed_data'] = ret_f['reconstructed_data']
            results['b_reconstructed_data'] = ret_b['reconstructed_data']

        return results
