"""
The implementation of the MLPs for ImputeFormer :cite:`nie2024imputeformer`
"""

# Created by Tong Nie <nietong@tongji.edu.cn> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn


class Dense(nn.Module):
    """A simple fully-connected layer."""

    def __init__(self, input_size, output_size, dropout=0.0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    """
    Simple Multi-layer Perceptron encoder with optional linear readout.
    """

    def __init__(self, input_size, hidden_size, output_size=None, n_layers=1, dropout=0.0):
        super().__init__()

        layers = [
            Dense(
                input_size=input_size if i == 0 else hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter("readout", None)

    def forward(self, x, u=None):
        """"""
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out
