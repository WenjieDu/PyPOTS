"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn

from .layers import FeatureRegression, Decay, Decay_obs, PositionalEncoding, Conv1dWithInit, TorchTransformerEncoder
from ..loss import Criterion, MAE


class BackboneCSAI(nn.Module):
    """
    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    medians_tensor :
        tensor of median values for features, used to adjust decayed observations

    temp_decay_h :
        the temporal decay module to decay the hidden state of the GRU

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist :
        the temporal-regression module that projects the GRU hidden state into the raw feature space

    feat_reg_v :
        the feature-regression module used for feature-based estimation

    weight_combine :
        the module that generates the weight to combine history regression and feature regression

    weighted_obs :
        the decay module that computes weighted decay based on observed data and deltas

    gru :
        the GRU cell that models temporal data for imputation

    pos_encoder :
        the positional encoding module that adds temporal information to the sequence data

    input_projection :
        the convolutional module used to project input features into a higher-dimensional space

    output_projection1 :
        the convolutional module used to project the output from the Transformer layer

    output_projection2 :
        the final convolutional module used to generate the hidden state from the time-layer's output

    time_layer :
        the Transformer encoder layer used to model complex temporal dependencies within the sequence

    device :
        the device (CPU/GPU) used for model computations

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    step_channels :
        number of channels for each step in the sequence

    """

    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        step_channels,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()

        self.n_steps = n_steps
        self.step_channels = step_channels
        self.input_size = n_features
        self.hidden_size = rnn_hidden_size
        self.training_loss = training_loss

        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag=False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag=True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.weighted_obs = Decay_obs(self.input_size, self.input_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)

        self.pos_encoder = PositionalEncoding(self.step_channels)
        self.input_projection = Conv1dWithInit(self.input_size, self.step_channels, 1)
        self.output_projection1 = Conv1dWithInit(self.step_channels, self.hidden_size, 1)
        self.output_projection2 = Conv1dWithInit(self.n_steps * 2, 1, 1)
        self.time_layer = TorchTransformerEncoder(channels=self.step_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, mask, deltas, last_obs, intervals, h=None):

        if intervals is not None:
            medians_tensor = torch.tensor(list(intervals.values())).float()
        else:
            medians_tensor = torch.zeros(x.shape[2]).float()

        medians = medians_tensor.unsqueeze(0).repeat(x.shape[0], 1).to(x.device)

        decay_factor = self.weighted_obs(deltas - medians.unsqueeze(1))

        if h is None:
            data_last_obs = self.input_projection(last_obs.permute(0, 2, 1)).permute(0, 2, 1)
            data_decay_factor = self.input_projection(decay_factor.permute(0, 2, 1)).permute(0, 2, 1)

            data_last_obs = self.pos_encoder(data_last_obs.permute(1, 0, 2)).permute(1, 0, 2)
            data_decay_factor = self.pos_encoder(data_decay_factor.permute(1, 0, 2)).permute(1, 0, 2)

            data = torch.cat([data_last_obs, data_decay_factor], dim=1)

            data = self.time_layer(data)
            data = self.output_projection1(data.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.output_projection2(data).squeeze()

        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        reconstruction = []
        for t in range(self.n_steps):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h

            # history based estimation
            x_h = self.hist(h)

            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # feature based estimation
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)

            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            # x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)
            x_loss += self.training_loss(x_comb_t, x_t, m_t)

            # Final Imputation Estimates
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)

            h = self.gru(input_t, h)
            Hiddens.append(h.unsqueeze(dim=1))
            reconstruction.append(x_comb_t.unsqueeze(dim=1))

        reconstruction = torch.cat(reconstruction, dim=1)

        return x_imp, reconstruction, h, x_loss


class BackboneBCSAI(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        rnn_hidden_size,
        step_channels,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()

        self.model_f = BackboneCSAI(n_steps, n_features, rnn_hidden_size, step_channels, training_loss)
        self.model_b = BackboneCSAI(n_steps, n_features, rnn_hidden_size, step_channels, training_loss)

    def forward(self, xdata):

        # Fetching forward data from xdata
        x = xdata["forward"]["X"]
        m = xdata["forward"]["missing_mask"]
        d_f = xdata["forward"]["deltas"]
        last_obs_f = xdata["forward"]["last_obs"]

        # Fetching backward data from xdata
        x_b = xdata["backward"]["X"]
        m_b = xdata["backward"]["missing_mask"]
        d_b = xdata["backward"]["deltas"]
        last_obs_b = xdata["backward"]["last_obs"]

        intervals = xdata["intervals"]

        # Call forward model
        (
            f_imputed_data,
            f_reconstruction,
            f_hidden_states,
            f_reconstruction_loss,
        ) = self.model_f(x, m, d_f, last_obs_f, intervals)

        # Call backward model
        (
            b_imputed_data,
            b_reconstruction,
            b_hidden_states,
            b_reconstruction_loss,
        ) = self.model_b(x_b, m_b, d_b, last_obs_b, intervals)

        # Averaging the imputations and prediction
        x_imp = (f_imputed_data + b_imputed_data.flip(dims=[1])) / 2
        imputed_data = (x * m) + ((1 - m) * x_imp)

        # average consistency loss
        consistency_loss = torch.abs(f_imputed_data - b_imputed_data.flip(dims=[1])).mean() * 1e-1

        # Merge the regression loss
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
