"""

"""

# Created by Joseph Arul Raj <joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class Decay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.W = None
        self.b = None
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class Decay_obs(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, delta_diff):
        # When delta_diff is negative, weight tends to 1.
        # When delta_diff is positive, weight tends to 0.
        sign = torch.sign(delta_diff)
        weight_diff = self.linear(delta_diff)
        # weight_diff can be either positive or negative for each delta_diff
        positive_part = F.relu(weight_diff)
        negative_part = F.relu(-weight_diff)
        weight_diff = positive_part + negative_part
        weight_diff = sign * weight_diff
        # Using a tanh activation to squeeze values between -1 and 1
        weight_diff = torch.tanh(weight_diff)
        # This will move the weight values towards 1 if delta_diff is negative
        # and towards 0 if delta_diff is positive
        weight = 0.5 * (1 - weight_diff)

        return weight


class TorchTransformerEncoder(nn.Module):
    def __init__(self, heads=8, layers=1, channels=64):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class Conv1dWithInit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
