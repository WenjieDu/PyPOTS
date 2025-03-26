"""
The implementation of the modules for Transformer :cite:`vaswani2017Transformer`

Notes
-----
This implementation is inspired by the official one https://github.com/WenjieDu/SAITS,
and https://github.com/jadore801120/attention-is-all-you-need-pytorch.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from abc import abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionOperator(nn.Module):
    """
    The abstract class for all attention layers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ScaledDotProductAttention(AttentionOperator):
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
        assert temperature > 0, f"temperature should be positive but got {temperature}"
        assert attn_dropout >= 0, f"dropout rate should be non-negative but got {attn_dropout}"
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
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
        # q, k, v all have 4 dimensions [batch_size, n_steps, n_heads, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        # transpose for attention dot product: [batch_size, n_heads, n_steps, d_k or d_v]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # dot product q with k.T to obtain similarity
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # apply masking on the attention map, this is optional
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -torch.inf)

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
    attn_opt:
        The attention operator, e.g. the self-attention proposed in Transformer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    """

    def __init__(
        self,
        attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.attention_operator = attn_opt
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        **kwargs,
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
        # the shapes of q, k, v are the same [batch_size, n_steps, d_model]

        batch_size, q_len = q.size(0), q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        # now separate the last dimension of q, k, v into different heads -> [batch_size, n_steps, n_heads, d_k or d_v]
        q = self.w_qs(q).view(batch_size, q_len, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, k_len, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, v_len, self.n_heads, self.d_v)
        # for generalization, we don't do transposing here but leave it for the attention operator if necessary

        if attn_mask is not None:
            # broadcasting on the head axis
            attn_mask = attn_mask.unsqueeze(1)

        v, attn_weights = self.attention_operator(q, k, v, attn_mask, **kwargs)

        # transpose back -> [batch_size, n_steps, n_heads, d_v]
        # then merge the last two dimensions to combine all the heads -> [batch_size, n_steps, n_heads*d_v]
        v = v.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        v = self.fc(v)

        return v, attn_weights
