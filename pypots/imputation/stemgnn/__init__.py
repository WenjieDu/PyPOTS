"""
The package of the partially-observed time-series imputation model StemGNN.

Refer to the paper
`Defu Cao, Yujing Wang, Juanyong Duan, Ce Zhang, Xia Zhu,
Congrui Huang, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, Qi Zhang.
"Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting".
In Advances in Neural Information Processing Systems, 2020.
<https://proceedings.neurips.cc/paper_files/paper/2020/file/cdf6581cb7aca4b7e19ef136c6e601a5-Paper.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/microsoft/StemGNN

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import StemGNN

__all__ = [
    "StemGNN",
]
