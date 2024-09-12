"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.fft
import torch.nn as nn


class RevIN(nn.Module):
    """RevIN: Reversible Inference Network.

    Parameters
    ----------
    n_features :
        the number of features or channels

    eps :
        a value added for numerical stability

    affine :
        if True, RevIN has learnable affine parameters

    """

    def __init__(
        self,
        n_features: int,
        eps: float = 1e-9,
        affine: bool = True,
    ):
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

        # calculate mean and stdev
        if missing_mask is None:
            # original implementation
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)
        else:
            # pypots implementation for POTS data
            missing_sum = torch.sum(missing_mask == 1, dim=dim2reduce, keepdim=True) + self.eps
            mean = torch.sum(x, dim=dim2reduce, keepdim=True) / missing_sum
            x_enc = x.masked_fill(missing_mask == 0, 0)
            variance = torch.sum(x_enc * x_enc, dim=dim2reduce, keepdim=True) + self.eps
            stdev = torch.sqrt(variance / missing_sum)

        # detach mean and stdev to avoid backpropagation
        self.mean = mean.detach()
        self.stdev = stdev.detach()
        # normalize the input
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            # apply affine transformation
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # reverse affine transformation
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        # denormalize the input
        x = x * self.stdev
        x = x + self.mean
        return x
