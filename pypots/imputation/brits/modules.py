"""
The implementation of the modules for BRITS.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS. The bugs in the original implementation
are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class FeatureRegression(nn.Module):
    """The module used to capture the correlation between features for imputation.

    Attributes
    ----------
    W : tensor
        The weights (parameters) of the module.

    b : tensor
        The bias of the module.

    m (buffer) : tensor
        The mask matrix, a squire matrix with diagonal entries all zeroes while left parts all ones.
        It is applied to the weight matrix to mask out the estimation contributions from features themselves.
        It is used to help enhance the imputation performance of the network.

    Parameters
    ----------
    input_size : the feature dimension of the input
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        x : tensor,
            the input for processing

        Returns
        -------
        output: tensor,
            the processed result containing imputation from feature regression

        """
        output = F.linear(x, self.W * Variable(self.m), self.b)
        return output


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the original paper.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing
    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        delta : tensor, shape [batch size, sequence length, feature number]
            The time gaps.

        Returns
        -------
        gamma : array-like, same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
