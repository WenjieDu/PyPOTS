"""
The package including the modules of FiLM.

Refer to the paper
`Zhou, Tian, Ziqing Ma, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, and Rong Jin.
"Film: Frequency improved legendre memory model for long-term time series forecasting."
In Advances in Neural Information Processing Systems 35 (2022): 12677-12690.
<https://proceedings.neurips.cc/paper_files/paper/2022/file/524ef58c2bd075775861234266e5e020-Paper-Conference.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/tianzhou2011/FiLM

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .backbone import BackboneFiLM
from .layers import HiPPO_LegT, SpectralConv1d

__all__ = [
    "HiPPO_LegT",
    "SpectralConv1d",
    "BackboneFiLM",
]
