"""
The package including the modules of Autoformer.

Refer to the paper
`Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long.
Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting.
In Advances in Neural Information Processing Systems, volume 34, pages 22419–22430. Curran Associates, Inc., 2021.
<https://proceedings.neurips.cc/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .auto_encoder import AutoformerEncoder
from .layers import (
    AutoCorrelation,
    AutoCorrelationLayer,
    SeasonalLayerNorm,
    MovingAvgBlock,
    SeriesDecompositionBlock,
    AutoformerEncoderLayer,
    AutoformerDecoderLayer,
)

__all__ = [
    "AutoCorrelation",
    "AutoCorrelationLayer",
    "SeasonalLayerNorm",
    "MovingAvgBlock",
    "SeriesDecompositionBlock",
    "AutoformerEncoderLayer",
    "AutoformerDecoderLayer",
    "AutoformerEncoder",
]
