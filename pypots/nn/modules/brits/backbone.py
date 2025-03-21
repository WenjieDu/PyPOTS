"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple

import torch
import torch.nn as nn

from .layers import FeatureRegression
from ..grud.layers import TemporalDecay
from ..loss import Criterion, MAE


class BackboneRITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

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


    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.training_loss = training_loss

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def forward(self, inputs: dict, direction: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from `inputs`.

        Returns
        -------
        imputed_data :
            Input data with missing parts imputed. Shape of [batch size, sequence length, feature number].

        estimations :
            Reconstructed data. Shape of [batch size, sequence length, feature number].

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        X = inputs[direction]["X"]  # feature values
        missing_mask = inputs[direction]["missing_mask"]  # mask marks missing part in X
        deltas = inputs[direction]["deltas"]  # time-gap values

        device = X.device

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        cell_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(device)

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = X[:, t, :]  # values
            m = missing_mask[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += self.training_loss(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += self.training_loss(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += self.training_loss(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(inputs, (hidden_states, cell_states))

        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        reconstruction = torch.cat(estimations, dim=1)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        return imputed_data, reconstruction, hidden_states, reconstruction_loss


class BackboneBRITS(nn.Module):
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
        training_loss: Criterion = MAE(),
    ):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = BackboneRITS(n_steps, n_features, rnn_hidden_size, training_loss)
        self.rits_b = BackboneRITS(n_steps, n_features, rnn_hidden_size, training_loss)

    @staticmethod
    def _get_consistency_loss(pred_f: torch.Tensor, pred_b: torch.Tensor) -> torch.Tensor:
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
    def _reverse(ret: Tuple) -> Tuple:
        """Reverse the array values on the time dimension in the given dictionary."""

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
            return tensor_.index_select(1, indices)

        collector = []
        for value in ret:
            collector.append(reverse_tensor(value))

        return tuple(collector)

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, ...]:
        # Results from the forward RITS.
        (
            f_imputed_data,
            f_reconstruction,
            f_hidden_states,
            f_reconstruction_loss,
        ) = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        (
            b_imputed_data,
            b_reconstruction,
            b_hidden_states,
            b_reconstruction_loss,
        ) = self._reverse(self.rits_b(inputs, "backward"))

        imputed_data = (f_imputed_data + b_imputed_data) / 2
        consistency_loss = self._get_consistency_loss(f_imputed_data, b_imputed_data)
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss

        return (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        )
