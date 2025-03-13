"""
The package of the partially-observed time-series imputation model TRMF.

Refer to the paper
`Hsiang-Fu Yu, Nikhil Rao, and Inderjit S. Dhillon.
"Temporal regularized matrix factorization for high-dimensional time series prediction."
In NeurIPS 2016.
<https://papers.nips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf>`_

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import TRMF

__all__ = [
    "TRMF",
]
