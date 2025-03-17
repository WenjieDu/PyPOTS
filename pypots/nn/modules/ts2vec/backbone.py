"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .layers import DilatedConvEncoder
from .utils import (
    torch_pad_nan,
    generate_binomial_mask,
    generate_continuous_mask,
)

MASK_MODES = ["binomial", "continuous", "all_true", "all_false", "mask_last"]


class TS2VecEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_pred_features: int,
        d_hidden: int,
        n_layers: int,
        mask_mode: str = "binomial",
    ):
        super().__init__()
        assert mask_mode in MASK_MODES, f"mask_mode should be one of {MASK_MODES}"

        self.n_features = n_features
        self.n_pred_features = n_pred_features
        self.d_hidden = d_hidden
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(n_features, d_hidden)
        self.feature_extractor = DilatedConvEncoder(d_hidden, [d_hidden] * n_layers + [n_pred_features], kernel_size=3)
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x n_features
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.forward(x, mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2), kernel_size=encoding_window, stride=1, padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2), kernel_size=(1 << (p + 1)) + 1, stride=1, padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[str] = None,
        encoding_window: Optional[int] = None,
        causal: bool = False,
        sliding_length: Optional[int] = None,
        sliding_padding: int = 0,
    ) -> torch.Tensor:
        """Compute representations using the trained model.

        Parameters
        ----------
        x:
            This should have a shape of (n_samples, n_steps, n_features). All missing data should be set to NaN.

        mask:
            The mask used by encoder can be specified with this parameter.
            This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.

        encoding_window:
            When this param is specified, the computed representation would the max pooling over this window.
            This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.

        causal:
            When this param is set to True, the future information would not be encoded into representation of
            each timestamp.

        sliding_length:
            The length of sliding window. When this param is specified,
            a sliding inference would be applied on the time series.

        sliding_padding:
            This param specifies the contextual data length used for inference every sliding windows.

        Returns
        -------
            repr: The representations for data.

        """

        n_samples, n_steps, _ = x.shape

        if sliding_length is not None:
            reprs = []
            for i in range(0, n_steps, sliding_length):
                left = i - sliding_padding
                right = i + sliding_length + (sliding_padding if not causal else 0)
                x_sliding = torch_pad_nan(
                    x[:, max(left, 0) : min(right, n_steps)],
                    left=-left if left < 0 else 0,
                    right=right - n_steps if right > n_steps else 0,
                    dim=1,
                )
                out = self._eval_with_pooling(
                    x_sliding,
                    mask,
                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                    encoding_window=encoding_window,
                )
                reprs.append(out)

            reprs = torch.cat(reprs, dim=1)
            if encoding_window == "full_series":
                reprs = F.max_pool1d(
                    reprs.transpose(1, 2).contiguous(),
                    kernel_size=reprs.size(1),
                ).squeeze(1)
        else:
            reprs = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
            if encoding_window == "full_series":
                reprs = reprs.squeeze(1)

        return reprs
