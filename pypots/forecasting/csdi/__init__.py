"""
The implementation of CSDI for the partially-observed time-series forecasting task.

Refer to the paper
`Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon.
CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.
In NeurIPS, 2021.
<https://proceedings.neurips.cc/paper_files/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf>`_

Notes
-----
This implementation is inspired by the official one the official implementation https://github.com/ermongroup/CSDI

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import CSDI

__all__ = [
    "CSDI",
]
