"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .layers import get_mask, refer_points, Bottleneck_Construct
from ..transformer.attention import ScaledDotProductAttention
from ..transformer.layers import TransformerEncoderLayer


class PyraformerEncoder(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        window_size: list,
        inner_size: int,
    ):
        super().__init__()

        d_bottleneck = d_model // 4
        d_k = d_v = d_model // n_heads

        self.mask, self.all_size = get_mask(n_steps, window_size, inner_size)
        self.indexes = refer_points(self.all_size, window_size)
        self.layer_stack = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )  # in the official code, they only use the naive pyramid attention
        self.conv_layers = Bottleneck_Construct(d_model, window_size, d_bottleneck)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:

        mask = self.mask.repeat(len(x), 1, 1).to(x.device)
        x = self.conv_layers(x)

        attn_weights_collector = []
        for layer in self.layer_stack:
            x, attn_weights = layer(x, mask)
            attn_weights_collector.append(attn_weights)

        indexes = self.indexes.repeat(x.size(0), 1, 1, x.size(2)).to(x.device)
        indexes = indexes.view(x.size(0), -1, x.size(2))
        all_enc = torch.gather(x, 1, indexes)
        enc_output = all_enc.view(x.size(0), self.all_size[0], -1)

        return enc_output, attn_weights_collector
