"""
The package of the partially-observed time-series forecasting model BTTF.

Refer to the paper
`Xinyu Chen and Lijun Sun.
Bayesian Temporal Factorization for Multidimensional Time Series Prediction.
IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–1, 2021.
<https://arxiv.org/pdf/1910.06366>`_

Notes
-----
This numpy implementation is the same with the official one from https://github.com/xinychen/transdim.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import BTTF

__all__ = [
    "BTTF",
]
