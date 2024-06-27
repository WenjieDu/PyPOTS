"""
The package including the modules of Non-stationary Transformer.

Refer to the paper
`Yong Liu, Haixu Wu, Jianmin Wang, Mingsheng Long.
Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting.
Advances in Neural Information Processing Systems 35 (2022): 9881-9893.
<https://proceedings.neurips.cc/paper_files/paper/2022/file/4054556fcaa934b0bf76da52cf4f92cb-Paper-Conference.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/thuml/Nonstationary_Transformers

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .autoencoder import NonstationaryTransformerEncoder
from .layers import DeStationaryAttention, Projector

__all__ = [
    "NonstationaryTransformerEncoder",
    "DeStationaryAttention",
    "Projector",
]
