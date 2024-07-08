"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch.nn as nn

from .layers import TemporalBlock


class BackboneTCN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=2,
        dropout=0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
