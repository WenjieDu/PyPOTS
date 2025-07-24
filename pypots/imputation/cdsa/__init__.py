"""
The package of the partially-observed time-series imputation model CDSA.

Refer to the paper "Ma, J., Shou, Z., Zareian, A., Mansour, H., Vetro, A., & Chang, S. F. (2019).
CDSA: cross-dimensional self-attention for multivariate, geo-tagged time series imputation.
arXiv preprint arXiv:1905.09904."

"""

# Created by Weixuan Chen <wx_chan@qq.com> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import CDSA

__all__ = [
    "CDSA",
]
