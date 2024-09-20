"""

"""

# Created by Tianxiang Zhan <zhantianxianguestc@hotmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class EvidenceMachineKernel(nn.Module):
    def __init__(self, C, F):
        super().__init__()
        self.C = C
        self.F = 2**F
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))

    def forward(self, x):
        x = torch.einsum("btc,cf->btcf", x, self.C_weight) + self.C_bias
        return x
