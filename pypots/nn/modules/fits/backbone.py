"""

"""

# Created by Wenjie Du <wdu@time-series.ai>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class BackboneFITS(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        cut_freq: int,
        individual: bool,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.individual = individual

        self.dominance_freq = cut_freq
        self.length_ratio = (n_steps + n_pred_steps) / n_steps

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.n_features):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)
                )
        else:
            # complex layer for frequency upsampling
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                torch.cfloat
            )

    def forward(self, x):
        low_specx = torch.fft.rfft(x, dim=1)
        assert low_specx.size(1) >= self.dominance_freq, (
            f"The sequence length after FFT {low_specx.size(1)} is less than the cut frequency {self.dominance_freq}. "
            f"Please check the input sequence length, or decrease the cut frequency."
        )
        low_specx[:, self.dominance_freq :] = 0  # LPF
        low_specx = low_specx[:, 0 : self.dominance_freq, :]  # LPF

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype,
            ).to(low_specx.device)
            for i in range(self.n_features):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.n_steps + self.n_pred_steps) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype,
        ).to(low_specxy_.device)
        low_specxy[:, 0 : low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compensation for the length change

        return low_xy
