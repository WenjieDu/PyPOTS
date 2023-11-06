"""
The implementation of the modules for Transformer :cite:`vaswani2017Transformer`

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS,
and https://github.com/jadore801120/attention-is-all-you-need-pytorch.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention.

    Parameters
    ----------
    temperature:
        The temperature for scaling.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        assert temperature > 0, "temperature should be positive"
        assert attn_dropout >= 0, "dropout rate should be non-negative"
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the scaled dot-product attention.

        Parameters
        ----------
        q:
            Query tensor.
        k:
            Key tensor.
        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        output:
            The result of Value multiplied with the scaled dot-product attention map.

        attn:
            The scaled dot-product attention map.

        """
        # q, k, v all have 4 dimensions [batch_size, n_heads, n_steps, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        # dot product q with k.T to obtain similarity
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # apply masking on the attention map, this is optional
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # compute attention score [0, 1], then apply dropout
        attn = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # multiply the score with v
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Transformer multi-head attention module.

    Parameters
    ----------
    n_heads:
        The number of heads in multi-head attention.

    d_model:
        The dimension of the input tensor.

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
        n_heads: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the multi-head attention module.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        v:
            The output of the multi-head attention layer.

        attn_weights:
            The attention map.

        """
        # the input q, k, v currently have 3 dimensions [batch_size, n_steps, d_tensor]
        # d_tensor could be n_heads*d_k, n_heads*d_v

        # keep useful variables
        batch_size, n_steps = q.size(0), q.size(1)
        residual = q

        # now separate the last dimension of q, k, v into different heads -> [batch_size, n_steps, n_heads, d_k or d_v]
        q = self.w_qs(q).view(batch_size, n_steps, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, n_steps, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, n_steps, self.n_heads, self.d_v)

        # transpose for self-attention calculation -> [batch_size, n_steps, d_k or d_v, n_heads]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # broadcasting on the head axis
            attn_mask = attn_mask.unsqueeze(1)

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # transpose back -> [batch_size, n_steps, n_heads, d_v]
        # then merge the last two dimensions to combine all the heads -> [batch_size, n_steps, n_heads*d_v]
        v = v.transpose(1, 2).contiguous().view(batch_size, n_steps, -1)
        v = self.fc(v)

        # apply dropout and residual connection
        v = self.dropout(v)
        v += residual

        # apply layer-norm
        v = self.layer_norm(v)

        return v, attn_weights


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


class DecoderLayer(nn.Module):
    """Transformer decoder layer.

    Parameters
    ----------
    d_model:
        The dimension of the input tensor.

    d_inner:
        The dimension of the hidden layer.

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
        self.enc_attn = MultiHeadAttention(
            n_heads, d_model, d_k, d_v, dropout, attn_dropout
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(
        self,
        dec_input: torch.Tensor,
        enc_output: torch.Tensor,
        slf_attn_mask: Optional[torch.Tensor] = None,
        dec_enc_attn_mask: Optional[torch.Tensor] = None,
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
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


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
        self.position_enc = PositionalEncoding(d_model, n_position=n_steps)
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
        self.position_enc = PositionalEncoding(d_model, n_position=n_steps)
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


class PositionalEncoding(nn.Module):
    """Positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_position:
        The number of positions.

    """

    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        x:
            Output tensor, the input tensor with the positional encoding added.

        """
        return x + self.pos_table[:, : x.size(1)].clone().detach()
