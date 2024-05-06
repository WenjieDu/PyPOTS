"""
The package including the modules of Informer.

Refer to the paper
`Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
Informer: Beyond efficient transformer for long sequence time-series forecasting.
In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 11106–11115, 2021.
<https://ojs.aaai.org/index.php/AAAI/article/download/17325/17132>`_

Notes
-----
This implementation is inspired by the official one https://github.com/zhouhaoyi/Informer2020

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .autoencoder import InformerEncoder, InformerDecoder
from .layers import ProbAttention, ConvLayer, InformerEncoderLayer, InformerDecoderLayer

__all__ = [
    "ProbAttention",
    "ConvLayer",
    "InformerEncoderLayer",
    "InformerDecoderLayer",
    "InformerEncoder",
    "InformerDecoder",
]
