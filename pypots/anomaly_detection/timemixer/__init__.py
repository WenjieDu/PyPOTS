"""
The package of the partially-observed time-series anomaly detection model TimeMixer.

Refer to the paper
`Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y. Zhang, and Jun Zhou.
"TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting".
In ICLR 2024.
<https://openreview.net/pdf?id=7oLshfEIC2>`_

Notes
-----
This implementation is inspired by the official one https://github.com/kwuking/TimeMixer

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause


from .model import TimeMixer

__all__ = [
    "TimeMixer",
]
