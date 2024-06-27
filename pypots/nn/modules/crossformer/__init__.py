"""
The package including the modules of Crossformer.

Refer to the paper
`Yunhao Zhang and Junchi Yan.
Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting.
In The 11th ICLR, 2023.
<https://openreview.net/pdf?id=vSVLM2j9eie>`_

Notes
-----
This implementation is inspired by the official one https://github.com/Thinklab-SJTU/Crossformer

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .autoencoder import CrossformerEncoder, CrossformerDecoder
from .layers import TwoStageAttentionLayer, ScaleBlock, CrossformerDecoderLayer

__all__ = [
    "CrossformerEncoder",
    "CrossformerDecoder",
    "TwoStageAttentionLayer",
    "ScaleBlock",
    "CrossformerDecoderLayer",
]
