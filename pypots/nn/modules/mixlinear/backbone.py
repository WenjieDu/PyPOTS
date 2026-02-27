"""
The backbone of MixLinear model.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneMixLinear(nn.Module):
    """The backbone of MixLinear :cite:`ma2026mixlinear`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the input sequence.

    n_features :
        The number of features (channels) in the input sequence.

    n_pred_steps :
        The number of steps to forecast.

    period_len :
        The segment length for the segment-based time-domain processing.
        Determines the granularity of local patterns captured.

    lpf :
        The number of low-frequency components to retain after FFT in the
        frequency-domain pathway (low-pass filter cutoff).

    alpha :
        The mixing weight between the time-domain and frequency-domain outputs.
        Final output = alpha * time_output + (1 - alpha) * frequency_output.

    rank :
        The rank of the low-rank spectral filter in the frequency pathway.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        period_len: int,
        lpf: int,
        alpha: float,
        rank: int = 2,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.period_len = period_len
        self.lpf = lpf
        self.alpha = alpha

        self.seg_num_x = math.ceil(n_steps / period_len)
        self.seg_num_y = math.ceil(n_pred_steps / period_len)

        assert self.lpf <= self.seg_num_x, (
            f"lpf ({lpf}) must be <= ceil(n_steps / period_len) = {self.seg_num_x}. "
            f"Please reduce lpf or period_len."
        )

        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seg_num_x))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.seg_num_y))

        # Time-domain pathway: factorized linear layers
        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)

        # Local trend smoothing via 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=period_len + 1,
            stride=1,
            padding=period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        # Frequency-domain pathway: low-rank complex filters
        self.FLinear1 = nn.Linear(lpf, rank, bias=False).to(torch.cfloat)
        self.FLinear2 = nn.Linear(rank, self.seg_num_y, bias=False).to(torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of BackboneMixLinear.

        Parameters
        ----------
        x :
            Input tensor of shape (batch_size, n_steps, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_pred_steps, n_features).

        """
        batch_size = x.shape[0]

        # Instance normalization: subtract mean over time (b,s,c -> scalar per feature)
        seq_mean = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, n_features)
        x = (x - seq_mean).permute(0, 2, 1)  # (batch, n_features, n_steps)

        # Local trend smoothing with Conv1d
        x = (
            self.conv1d(x.reshape(-1, 1, self.n_steps)).reshape(-1, self.n_features, self.n_steps) + x
        )  # (batch, n_features, n_steps)

        # Pad to multiple of period_len if necessary
        pad_len = self.seg_num_x * self.period_len - self.n_steps
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # Reshape to segments: (batch, n_features, seg_num_x, period_len) -> permute -> (batch, n_features, period_len, seg_num_x)
        x = x.reshape(batch_size, self.n_features, self.seg_num_x, self.period_len).permute(0, 1, 3, 2)

        # ── Time domain ──────────────────────────────────────────────────────────
        # Pad seg dimension to sqrt_seg_num_x^2 for factorized 2-D linear
        x_o = F.pad(x, (0, self.sqrt_seg_num_x**2 - self.seg_num_x))
        # (batch, n_features, period_len, sqrt_seg_num_x^2)
        x_o = x_o.reshape(batch_size, self.n_features, self.period_len, self.sqrt_seg_num_x, self.sqrt_seg_num_x)
        # (batch, n_features, period_len, sqrt_seg_num_x, sqrt_seg_num_x)

        x_o = self.TLinear1(x_o).permute(0, 1, 2, 4, 3)
        # TLinear1 maps last dim sqrt_seg_num_x -> sqrt_seg_num_y
        # After: (batch, n_features, period_len, sqrt_seg_num_x, sqrt_seg_num_y)
        # After permute: (batch, n_features, period_len, sqrt_seg_num_y, sqrt_seg_num_x)

        x_t = self.TLinear2(x_o).permute(0, 1, 2, 4, 3)
        # After: (batch, n_features, period_len, sqrt_seg_num_y, sqrt_seg_num_y)
        # After permute: (batch, n_features, period_len, sqrt_seg_num_y, sqrt_seg_num_y)

        x_t = (
            x_t.reshape(batch_size, self.n_features, self.period_len, -1)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, self.n_features, -1)
            .permute(0, 2, 1)
        )
        # Final: (batch, sqrt_seg_num_y^2 * period_len, n_features)

        # ── Frequency domain ─────────────────────────────────────────────────────
        x_fft = torch.fft.fft(x, dim=3)[:, :, :, : self.lpf]
        # (batch, n_features, period_len, lpf)

        x_fft = self.FLinear1(x_fft)
        # (batch, n_features, period_len, rank)

        x_fft = self.FLinear2(x_fft).reshape(batch_size, self.n_features, self.period_len, -1)
        # (batch, n_features, period_len, seg_num_y)

        x_ifft_real = torch.fft.ifft(x_fft, dim=3).real
        # (batch, n_features, period_len, seg_num_y)

        x_f = x_ifft_real.permute(0, 1, 3, 2).reshape(batch_size, self.n_features, -1).permute(0, 2, 1)
        # Final: (batch, seg_num_y * period_len, n_features)

        # ── Mix ──────────────────────────────────────────────────────────────────
        output = (
            x_t[:, : self.n_pred_steps, :] * self.alpha
            + seq_mean
            + x_f[:, : self.n_pred_steps, :] * (1 - self.alpha)
        )

        return output
