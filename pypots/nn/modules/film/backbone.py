"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .layers import HiPPO_LegT, SpectralConv1d


class BackboneFiLM(nn.Module):
    def __init__(
        self,
        n_steps: int,
        in_channels: int,
        n_pred_steps: int,
        window_size: list,
        multiscale: list,
        modes1: int,
        ratio: float,
        mode_type: int,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.window_size = window_size
        self.multiscale = multiscale
        self.ratio = ratio

        self.affine_weight = nn.Parameter(torch.ones(1, 1, in_channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1.0 / n_pred_steps / i) for n in window_size for i in multiscale]
        )
        self.spec_conv_1 = nn.ModuleList(
            [
                SpectralConv1d(
                    in_channels=n,
                    out_channels=n,
                    seq_len=min(n_pred_steps, n_steps),
                    modes1=modes1,
                    ratio=ratio,
                    mode_type=mode_type,
                )
                for n in window_size
                for _ in range(len(multiscale))
            ]
        )
        self.mlp = nn.Linear(len(multiscale) * len(window_size), 1)

    def forward(self, X) -> torch.Tensor:
        x_enc = X * self.affine_weight + self.affine_bias

        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.n_pred_steps
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.n_steps >= self.n_pred_steps:
                x_dec_c = out1.transpose(2, 3)[:, :, self.n_pred_steps - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.n_pred_steps :, :].T
            x_decs.append(x_dec)

        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        return x_dec
