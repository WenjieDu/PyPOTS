"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause
#
# Portions of this file are adapted from the following project(s), redistributed under
# their original license terms reproduced below:
#   - AROMA: https://github.com/LouisSerrano/aroma
#
# ---- AROMA (https://github.com/LouisSerrano/aroma) ----
#
# MIT License
#
# Copyright (c) 2024 LouisSerrano
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
from torch import nn

from .nerf import NeRFEncoding


class FourierPositionalEmbedding(nn.Module):
    """Fourier Feature layer."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_freq: int = 32,
        max_freq_log2: int = 5,
        input_dim: int = 2,
        base_freq: int = 2,
        use_relu: bool = True,
    ) -> None:

        super().__init__()

        self.nerf_embedder = NeRFEncoding(
            num_freq=num_freq,
            max_freq_log2=max_freq_log2,
            min_freq_log2=0,
            input_dim=input_dim,
            base_freq=base_freq,
            log_sampling=False,
            include_input=True,
        )

        self.linear = nn.Linear(self.nerf_embedder.out_dim, hidden_dim)
        self.use_relu = use_relu

    def forward(self, coords: torch.Tensor) -> torch.Tensor:

        x = self.nerf_embedder(coords)
        if self.use_relu:
            x = torch.relu(self.linear(x))
        else:
            x = self.linear(x)  # try without relu

        return x
