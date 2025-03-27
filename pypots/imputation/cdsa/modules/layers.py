# Created by Weixuan Chen <wx_chan@qq.com> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple, Optional

import torch
import torch.nn as nn

from .submodules import MultiHeadCDSA
from ....nn.modules.transformer import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """CDSA encoder layer.

    Parameters
    ----------
    d_model:
        The dimension of the input tensor.

    d_inner:
        The dimension of the hidden layer.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor on time-series.

    d_v:
        The dimension of the value tensor,
        The dimension of the key and query tensor on feature.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadCDSA(n_heads, d_model, d_k, d_v, dropout, attn_dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(
        self,
        enc_input: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the encoder layer.

        Parameters
        ----------
        enc_input:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights:
            The attention map.

        """
        enc_output, attn_weights = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            attn_mask=src_mask,
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights
