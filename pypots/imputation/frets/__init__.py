"""
The package of the partially-observed time-series imputation model FreTS.

Refer to the paper
`Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, and Zhendong Niu.
"Frequency-domain MLPs are More Effective Learners in Time Series Forecasting."
Advances in Neural Information Processing Systems 36 (2024).
<https://proceedings.neurips.cc/paper_files/paper/2023/file/f1d16af76939f476b5f040fd1398c0a3-Paper-Conference.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/aikunyi/FreTS

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import FreTS

__all__ = [
    "FreTS",
]
