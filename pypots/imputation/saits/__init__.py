"""
The package of the partially-observed time-series imputation model SAITS.

Refer to the paper
`Wenjie Du, David Cote, and Yan Liu.
SAITS: Self-Attention-based Imputation for Time Series.
Expert Systems with Applications, 219:119619, 2023.
<https://arxiv.org/pdf/2202.08516>`_

Notes
-----
This implementation is inspired by the official one https://github.com/WenjieDu/SAITS

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import SAITS

__all__ = [
    "SAITS",
]
