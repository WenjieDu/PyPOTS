"""
The package including the modules of TRMF.

Refer to the paper
Yu, H. F., Rao, N., & Dhillon, I. S.
Temporal regularized matrix factorization for high-dimensional time series prediction. 
In Advances in neural information processing systems 2016.
<https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf>`_

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import TRMF

__all__ = [
    "TRMF",
]
