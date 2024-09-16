"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .layers import DeStationaryAttention
from ..transformer.layers import TransformerEncoderLayer


class NonstationaryTransformerEncoder(nn.Module):
    """NonstationaryTransformer encoder.
    Its arch is the same with the original Transformer encoder,
    but the attention operator is replaced by the DeStationaryAttention.

    Parameters
    ----------
    n_layers:
        The number of layers in the encoder.

    d_model:
        The dimension of the module manipulation space.
        The input tensor will be projected to a space with d_model dimensions.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer in the feed-forward network.

    dropout:
        The dropout rate.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()

        self.enc_layer_stack = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    DeStationaryAttention(d_k**0.5, attn_dropout),
                    d_model,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Forward processing of the encoder.

        Parameters
        ----------
        x:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights_collector:
            A list containing the attention map from each encoder layer.

        """
        attn_weights_collector = []
        enc_output = x

        if src_mask is None:
            # triangular causal mask
            bz, n_steps, _ = x.shape
            mask_shape = [bz, n_steps, n_steps]
            src_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(x.device)

        for layer in self.enc_layer_stack:
            enc_output, attn_weights = layer(enc_output, src_mask, **kwargs)
            attn_weights_collector.append(attn_weights)

        return enc_output, attn_weights_collector
