# Created by Weixuan Chen <wx_chan@qq.com> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDimensionalSelfAttention(nn.Module):
    """Cross Dimensional Self-Attention

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
        q_time: torch.Tensor,
        k_time: torch.Tensor,
        q_feature: torch.Tensor,
        k_feature: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward processing of the scaled dot-product attention.

        Parameters
        ----------
        q_time:
            Query tensor on time-series.
        k_time:
            Key tensor on time-series.
        q_feature:
            Query tensor on feature.
        k_feature:
            Key tensor on feature.
        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        output:
            The result of Value multiplied with the scaled dot-product attention map.

        attn_time:
            The scaled dot-product attention map on time-series.

        attn_feature:
            The scaled dot-product attention map on feature.
        """
        # q_time, k_time, v all have 4 dimensions [batch_size, n_heads, n_steps, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        # dot product q with k.T to obtain similarity
        attn_time = torch.matmul(
            q_time / self.temperature, k_time.transpose(2, 3)
        )  # [batch_size, n_heads, n_steps, n_steps]
        attn_feature = torch.matmul(
            q_feature.transpose(2, 3) / self.temperature, k_feature
        )  # [batch_size, n_heads, d_v, d_v]

        # apply masking on the attention map, this is optional
        if attn_mask is not None:
            attn_time = attn_time.masked_fill(attn_mask == 0, -1e9)

        # compute attention score [0, 1], then apply dropout
        attn_time = F.softmax(attn_time, dim=-1)
        attn_feature = F.softmax(attn_feature, dim=-1)
        if self.dropout is not None:
            attn_time = self.dropout(attn_time)
            attn_feature = self.dropout(attn_feature)

        # multiply the score with v
        output = torch.matmul(torch.matmul(attn_time, v), attn_feature)

        return output, attn_time, attn_feature


class MultiHeadCDSA(nn.Module):
    """Multi-head Cross-Dimensional Self-Attention module.

    Parameters
    ----------
    n_heads:
        The number of heads in multi-head attention.

    d_model:
        The dimension of the input tensor.

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

        # w_qs, w_ks for Time Dimension
        self.w_qs_time = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks_time = nn.Linear(d_model, n_heads * d_k, bias=False)

        # w_qs, w_ks for Feature Dimension
        self.w_qs_feature = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.w_ks_feature = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.attention = CrossDimensionalSelfAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q_time: torch.Tensor,
        k_time: torch.Tensor,
        q_feature: torch.Tensor,
        k_feature: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward processing of the multi-head CDSA module.

        Parameters
        ----------
        q_time:
            Query tensor for Time-Dimension.

        k_time:
            Key tensor for Time-Dimension.

        q_feature:
            Query tensor for Feature-Dimension.

        k_feature:
            Key tensor for Feature-Dimension.

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
        batch_size, n_steps = q_time.size(0), q_time.size(1)
        residual = v

        # now separate the last dimension of q, k, v into different heads -> [batch_size, n_steps, n_heads, d_k or d_v]
        q_time = self.w_qs(q_time).view(batch_size, n_steps, self.n_heads, self.d_k)
        k_time = self.w_ks(k_time).view(batch_size, n_steps, self.n_heads, self.d_k)
        q_feature = self.w_qs(q_feature).view(
            batch_size, n_steps, self.n_heads, self.d_v
        )
        k_feature = self.w_ks(k_feature).view(
            batch_size, n_steps, self.n_heads, self.d_v
        )
        v = self.w_vs(v).view(batch_size, n_steps, self.n_heads, self.d_v)

        # transpose for self-attention calculation -> [batch_size, n_steps, d_k or d_v, n_heads]
        q_time, k_time, q_feature, k_feature, v = (
            q_time.transpose(1, 2),
            k_time.transpose(1, 2),
            q_feature.transpose(1, 2),
            k_feature.transpose(1, 2),
            v.transpose(1, 2),
        )

        if attn_mask is not None:
            # broadcasting on the head axis
            attn_mask = attn_mask.unsqueeze(1)

        v, attn_weights_time, attn_weights_feature = self.attention(
            q_time, k_time, q_feature, k_feature, v, attn_mask
        )

        # transpose back -> [batch_size, n_steps, n_heads, d_v]
        # then merge the last two dimensions to combine all the heads -> [batch_size, n_steps, n_heads*d_v]
        v = v.transpose(1, 2).contiguous().view(batch_size, n_steps, -1)
        v = self.fc(v)

        # apply dropout and residual connection
        v = self.dropout(v)
        v += residual

        # apply layer-norm
        v = self.layer_norm(v)

        return v, attn_weights_time, attn_weights_feature
