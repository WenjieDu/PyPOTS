"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.fft
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(
        self,
        n_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Parameters
        ----------
        n_features :
            the number of features or channels

        eps :
            a value added for numerical stability

        affine :
            if True, RevIN has learnable affine parameters

        """
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, missing_mask=None, mode: str = "norm"):
        if mode == "norm":
            x = self._normalize(x, missing_mask)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.n_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.n_features))

    def _normalize(self, x, missing_mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))

        if missing_mask is None:
            # original implementation
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = x - self.mean
            x = x / self.stdev
        else:
            # pypots implementation for POTS data
            missing_sum = (
                torch.sum(missing_mask == 1, dim=dim2reduce, keepdim=True) + 1e-9
            )
            self.mean = torch.sum(x, dim=dim2reduce, keepdim=True) / missing_sum
            x = x - self.mean
            x_enc = x.masked_fill(missing_mask == 0, 0)
            variance = torch.sum(x_enc * x_enc, dim=dim2reduce, keepdim=True) + 1e-9
            self.stdev = torch.sqrt(variance / missing_sum)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
