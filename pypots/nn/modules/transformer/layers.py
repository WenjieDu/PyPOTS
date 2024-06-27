"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, AttentionOperator


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


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    Parameters
    ----------
    attn_opt:
        The attention operator for the multi-head attention module in the encoder layer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(
        self,
        attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            attn_opt,
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        enc_input: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        **kwargs,
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
            **kwargs,
        )

        # apply dropout and residual connection
        enc_output = self.dropout(enc_output)
        enc_output += enc_input

        # apply layer-norm
        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    Parameters
    ----------
    slf_attn_opt:
        The attention operator for the multi-head attention module in the decoder layer.

    enc_attn_opt:
        The attention operator for the encoding multi-head attention module in the decoder layer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(
        self,
        slf_attn_opt: AttentionOperator,
        enc_attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            slf_attn_opt,
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.enc_attn = MultiHeadAttention(
            enc_attn_opt,
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        dec_input: torch.Tensor,
        enc_output: torch.Tensor,
        slf_attn_mask: Optional[torch.Tensor] = None,
        dec_enc_attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward processing of the decoder layer.

        Parameters
        ----------
        dec_input:
            Input tensor.

        enc_output:
            Output tensor from the encoder.

        slf_attn_mask:
            Masking tensor for the self-attention module.
            The shape should be [batch_size, n_heads, n_steps, n_steps].

        dec_enc_attn_mask:
            Masking tensor for the encoding attention module.
            The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        dec_output:
            Output tensor.

        dec_slf_attn:
            The self-attention map.

        dec_enc_attn:
            The encoding attention map.

        """
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input,
            dec_input,
            dec_input,
            attn_mask=slf_attn_mask,
            **kwargs,
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output,
            enc_output,
            enc_output,
            attn_mask=dec_enc_attn_mask,
            **kwargs,
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
