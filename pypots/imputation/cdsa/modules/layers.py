
# License: BSD-3-Clause

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .CrossDimAttention import MultiHeadAttention


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward network (FFN) in Transformer.

    Parameters
    ----------
    d_in:
        The dimension of the input tensor.

    d_hid:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid)
        self.linear_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the position-wise feed forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        x:
            Output tensor.
        """
        # save the original input for the later residual connection
        residual = x
        # the 1st linear processing and ReLU non-linear projection
        x = F.relu(self.linear_1(x))
        # the 2nd linear processing
        x = self.linear_2(x)
        # apply dropout
        x = self.dropout(x)
        # apply residual connection
        x += residual
        # apply layer-norm
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer.

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
        self.slf_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, attn_dropout
        )
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

