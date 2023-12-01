"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .layers import EncoderLayer, DecoderLayer
from .pos_enc import PositionalEncoding


class Encoder(nn.Module):
    """Transformer encoder.

    Parameters
    ----------
    n_layers:
        The number of layers in the encoder.

    n_steps:
        The number of time steps in the input tensor.

    n_features:
        The number of features in the input tensor.

    d_model:
        The dimension of the module manipulation space.
        The input tensor will be projected to a space with d_model dimensions.

    d_inner:
        The dimension of the hidden layer in the feed-forward network.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        self.embedding = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_steps)
        self.enc_layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Forward processing of the encoder.

        Parameters
        ----------
        x:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        return_attn_weights:
            Whether to return the attention map.

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights_collector:
            A list containing the attention map from each encoder layer.

        """
        x = self.embedding(x)
        enc_output = self.dropout(self.position_enc(x))
        attn_weights_collector = []

        for layer in self.enc_layer_stack:
            enc_output, attn_weights = layer(enc_output, src_mask)
            attn_weights_collector.append(attn_weights)

        if return_attn_weights:
            return enc_output, attn_weights_collector

        return enc_output


class Decoder(nn.Module):
    """Transformer decoder.

    Parameters
    ----------
    n_layers:
        The number of layers in the decoder.

    n_steps:
        The number of time steps in the input tensor.

    n_features:
        The number of features in the input tensor.

    d_model:
        The dimension of the module manipulation space.
        The input tensor will be projected to a space with d_model dimensions.

    d_inner:
        The dimension of the hidden layer in the feed-forward network.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_steps)
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        trg_seq: torch.Tensor,
        enc_output: torch.Tensor,
        trg_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list, list]]:
        """Forward processing of the decoder.

        Parameters
        ----------
        trg_seq:
            Input tensor.

        enc_output:
            Output tensor from the encoder.

        trg_mask:
            Masking tensor for the self-attention module.

        src_mask:
            Masking tensor for the encoding attention module.

        return_attn_weights:
            Whether to return the attention map.

        Returns
        -------
        dec_output:
            Output tensor.

        dec_slf_attn_collector:
            A list containing the self-attention map from each decoder layer.

        dec_enc_attn_collector:
            A list containing the encoding attention map from each decoder layer.

        """
        trg_seq = self.embedding(trg_seq)
        dec_output = self.dropout(self.position_enc(trg_seq))

        dec_slf_attn_collector = []
        dec_enc_attn_collector = []

        for layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = layer(
                dec_output,
                enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
            )
            dec_slf_attn_collector.append(dec_slf_attn)
            dec_enc_attn_collector.append(dec_enc_attn)

        if return_attn_weights:
            return dec_output, dec_slf_attn_collector, dec_enc_attn_collector

        return dec_output
