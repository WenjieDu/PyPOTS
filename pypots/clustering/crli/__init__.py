"""
The package of the partially-observed time-series clustering model CRLI.

Refer to the paper "Ma, Q., Chen, C., Li, S., & Cottrell, G. W. (2021).
Learning Representations for Incomplete Time Series Clustering. AAAI 2021."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import CRLI

__all__ = [
    "CRLI",
]
