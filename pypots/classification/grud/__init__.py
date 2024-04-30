"""
The package of the partially-observed time-series classification model GRU-D.

Refer to the paper
`Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
Recurrent Neural Networks for Multivariate Time Series with Missing Values.
Scientific Reports, 8(1):6085, April 2018.
<https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

Notes
-----
This implementation is inspired by the official one  https://github.com/PeterChe1990/GRU-D

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import GRUD

__all__ = [
    "GRUD",
]
