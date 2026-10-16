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

from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import nn


class MultiScaleNeRFEncoding(nn.Module):
    def __init__(
        self,
        num_freqs_per_scale: int,
        log_sampling: bool = True,
        include_input: bool = False,
        input_dim: int = 3,
        base_freq: int = 2,
        scales: Sequence[int] = [3, 4, 5],
        use_pi: bool = True,
        disjoint: bool = True,
    ):
        super().__init__()

        self.num_freqs_per_scale = num_freqs_per_scale
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.input_dim = input_dim
        self.base_freq = base_freq
        self.scales = scales
        self.use_pi = use_pi
        self.disjoint = disjoint

        # Initialize bands for each scale
        # self.bands = nn.ParameterDict(self.initialize_bands(scales))
        dict_bands = self.initialize_bands(scales)
        self.band_names = list(dict_bands.keys())
        [self.register_buffer(key, band) for key, band in dict_bands.items()]

        # Calculate output dimension
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim
        for _ in scales:
            self.out_dim += num_freqs_per_scale * input_dim * 2  # sin and cos for each frequency band

        if include_input:
            self.out_dim_per_scale = num_freqs_per_scale * input_dim * 2 + 1
        else:
            self.out_dim_per_scale = num_freqs_per_scale * input_dim * 2

    def initialize_bands(self, scales):
        s = [0] + scales
        bands = {}
        for j in range(len(s) - 1):
            if self.disjoint:
                start, end = s[j], s[j + 1]
            else:
                start, end = s[0], s[j + 1]

            if self.log_sampling:
                band = self.base_freq ** torch.linspace(start, end, steps=self.num_freqs_per_scale)
            else:
                band = torch.linspace(
                    self.base_freq**start,
                    self.base_freq**end,
                    steps=self.num_freqs_per_scale,
                )
            if self.use_pi:
                band = band * np.pi
            bands[f"{s[j + 1]}".replace(".", "_")] = band
            # bands[f"{s[j+1]}"] = nn.Parameter(band, requires_grad=False)

        return bands

    def forward(self, coords, with_batch=True):
        encoded_list = []

        # if self.include_input:
        #     encoded_list.append(coords)

        # for scale, bands in self.bands.items():
        for band_name in self.band_names:
            bands = getattr(self, band_name)
            b = coords.shape[0]
            N = coords.shape[1]
            winded = (coords[..., None] * bands[None, :]).reshape(b, N, -1)
            # .reshape(coords.shape[0], -1)
            if self.include_input:
                encoded_scale = torch.cat([coords, torch.sin(winded), torch.cos(winded)], dim=-1)
            else:
                encoded_scale = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            encoded_list.append(encoded_scale)
            # print('encoded_scale', encoded_scale.shape)

        return torch.stack(encoded_list, dim=-2)

    def name(self) -> str:
        return "Multiscale Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freqs_per_scale * len(self.scales),
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
            "Scales": self.scales,
        }


class NeRFEncoding(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""

    def __init__(
        self,
        num_freq: int,
        max_freq_log2: int,
        log_sampling: bool = True,
        include_input: bool = True,
        min_freq_log2: int = 0,
        input_dim: int = 3,
        base_freq: float | int = 2,
    ) -> None:
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """

        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.base_freq = base_freq

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            bands = self.base_freq ** torch.linspace(min_freq_log2, max_freq_log2, steps=num_freq)  # [num_freq,]
        else:
            bands = self.base_freq * torch.arange(min_freq_log2, num_freq, 1)

        bands = bands.to(dtype=torch.float32)  # [num_freq,]

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += bands.shape[0] * input_dim * 2
        self.register_buffer("bands", bands)
        # self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords: torch.Tensor, with_batch: bool = True) -> torch.Tensor:
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """

        if with_batch:
            N = coords.shape[0]
            winded = (coords[..., None, :] * self.bands[None, None, :, None]).reshape(
                N, coords.shape[1], coords.shape[-1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded * 2 * torch.pi), torch.cos(winded * 2 * torch.pi)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)

        else:
            N = coords.shape[0]
            winded = (coords[:, None] * self.bands[None, :, None]).reshape(N, coords.shape[1] * self.num_freq)
            encoded = torch.cat([torch.sin(winded * 2 * torch.pi), torch.cos(winded * 2 * torch.pi)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """A human readable name for the given wisp module."""
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
        }
