"""
The package of the partially-observed time-series classification model BRITS.

Refer to the paper
`Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li.
BRITS: Bidirectional recurrent imputation for time series.
In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.
<https://papers.nips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/caow13/BRITS.
The bugs in the original implementation are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import BRITS

__all__ = [
    "BRITS",
]
