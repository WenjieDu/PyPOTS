"""
The package of the partially-observed time-series anomaly detection model TimesNet.

Refer to the paper
`Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long.
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.
In ICLR, 2023.
<https://openreview.net/pdf?id=ju_Uqw384Oq>`_

Notes
-----
This implementation is inspired by the official one https://github.com/thuml/TimesNet

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause


from .model import TimesNet

__all__ = [
    "TimesNet",
]
