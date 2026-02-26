"""
The package including the modules of MixLinear.

Refer to the paper
`Aitian Ma, Dongsheng Luo, and Mo Sha.
MixLinear: Extreme Low-Resource Multivariate Time Series Forecasting with 0.1K Parameters.
In International Conference on Learning Representations (ICLR), 2026.
<https://arxiv.org/pdf/2410.02081.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/aitianma/MixLinear

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .backbone import BackboneMixLinear

__all__ = [
    "BackboneMixLinear",
]
