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
#   - perceiver-pytorch: https://github.com/lucidrains/perceiver-pytorch
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
#
# ---- perceiver-pytorch (https://github.com/lucidrains/perceiver-pytorch) ----
#
# MIT License
#
# Copyright (c) 2021 Phil Wang
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
import torch.nn.functional as F
from torch import nn

from ..utils import exists


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None) -> None:
        """
        z-norm over the last dim of the input (LayerNorm) before applying a given module.

        Args:
            dim (int): last dimension of the input (queries)
            fn (nn.Module): the module before which LayerNorm is applied,
                typically `Attention`, `MultiScaleAttention` or `FeedForward`
            context_dim (int): last dimension of the context inputs, if provided
        """

        super().__init__()

        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None  # type: ignore

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        # normalize inputs:
        x = self.norm(x)

        # normalize context if provided:
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        # apply module on normalized inputs:
        return self.fn(x, **kwargs)


class PreNormCross(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, k_dim: int | None = None, v_dim: int | None = None):
        """
        z-norm over the last dim of the input (LayerNorm) before applying a given module.

        Args:
            dim (int): last dimension of the input (queries)
            fn (nn.Module): the module before which LayerNorm is applied,
                typically `CrossAttention`
            k_dim (int): last dimension of the keys, if provided
            v_dim (int): last dimension of the values, if provided
        """

        super().__init__()

        self.fn = fn
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.norm_k = nn.LayerNorm(k_dim) if exists(k_dim) else None  # type: ignore
        self.norm_v = nn.LayerNorm(v_dim) if exists(v_dim) else None  # type: ignore
        assert not (exists(k_dim) ^ exists(v_dim))  # not xor

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        # normalize queries:
        x = self.norm(x)

        # normalize keys and values:
        if exists(self.norm_v):
            k = kwargs["k"]
            v = kwargs["v"]
            normed_k = self.norm_k(k)
            normed_v = self.norm_v(v)
            kwargs.update(k=normed_k, v=normed_v)

        # apply module over normalized inputs:
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
