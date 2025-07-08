"""
The package of the partially-observed time-series forecasting model MICN.

Refer to the paper
`Huiqiang Wang, Jian Peng, Feihu Huang, Jince Wang, Junhui Chen, and Yifei Xiao
"MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting".
In the Eleventh International Conference on Learning Representations, 2023.
<https://openreview.net/pdf?id=zt53IDUR1U>`_

Notes
-----
This implementation is inspired by the official one https://github.com/wanghq21/MICN

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import MICN

__all__ = [
    "MICN",
]
