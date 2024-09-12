"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Optional, Tuple

import torch
import torch.fft
import torch.nn as nn

from ..transformer.attention import AttentionOperator


class DeStationaryAttention(AttentionOperator):
    """De-stationary Attention"""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        k: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k, v all have 4 dimensions [batch_size, n_steps, n_heads, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        B, L, H, E = q.shape
        _, S, _, D = v.shape
        temperature = self.temperature or 1.0 / math.sqrt(E)

        tau, delta = kwargs["tau"], kwargs["delta"]
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", q, k) * tau + delta

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -torch.inf)

        attn = self.dropout(torch.softmax(temperature * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", attn, v)
        output = V.contiguous()

        return output, attn


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    """

    def __init__(
        self,
        d_in: int,
        n_steps: int,
        d_hidden: list,
        n_hidden_layers: int,
        d_output: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        assert (
            len(d_hidden) == n_hidden_layers
        ), f"The length of d_hidden should be equal to n_hidden_layers, but got {len(d_hidden)} and {n_hidden_layers}."

        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=n_steps,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        layers = [nn.Linear(2 * d_in, d_hidden[0]), nn.ReLU()]
        for i in range(n_hidden_layers - 1):
            layers += [nn.Linear(d_hidden[i], d_hidden[i + 1]), nn.ReLU()]

        layers += [nn.Linear(d_hidden[-1], d_output, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y
