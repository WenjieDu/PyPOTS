"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy import special as ss


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization="bilinear"):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        A, B = self.transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer("A", torch.Tensor(A))
        self.register_buffer("B", torch.Tensor(B))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer(
            "eval_matrix",
            torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T),
        )

    @staticmethod
    def transition(N):
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        return A, B

    def forward(self, inputs: torch.Tensor):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        device = inputs.device
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len,
        modes1,
        ratio=0.5,
        mode_type=0,
        # compression=0, # never used in the official implementation, hence deprecate it here
    ):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.ratio = ratio

        if mode_type == 0:
            self.modes2 = min(32, seq_len // 2)
            self.index = list(range(0, self.modes2))
        elif mode_type == 1:
            modes2 = modes1
            self.modes2 = min(modes2, seq_len // 2)
            self.index0 = list(range(0, int(ratio * min(seq_len // 2, modes2))))
            self.index1 = list(range(len(self.index0), self.modes2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[: min(seq_len // 2, self.modes2) - int(ratio * min(seq_len // 2, modes2))]
            self.index = self.index0 + self.index1
            self.index.sort()
        elif mode_type == 2:
            modes2 = modes1
            self.modes2 = min(modes2, seq_len // 2)
            self.index = list(range(0, seq_len // 2))
            np.random.shuffle(self.index)
            self.index = self.index[: self.modes2]

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat)
        )

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            B,
            H,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        if self.modes1 > 1000:
            for wi, i in enumerate(self.index):
                out_ft[:, :, :, i] = torch.einsum("bji,io->bjo", (x_ft[:, :, :, i], self.weights1[:, :, wi]))
        else:
            a = x_ft[:, :, :, : self.modes2]
            out_ft[:, :, :, : self.modes2] = torch.einsum("bjix,iox->bjox", a, self.weights1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
