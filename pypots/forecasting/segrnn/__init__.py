"""
The package of the partially-observed time-series forecasting model SegRNN.

Refer to the paper
`Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long.
SegRNN: Temporal 2D-Variation Modeling for General Time Series Analysis.
In ICLR, 2023.
<https://openreview.net/pdf?id=ju_Uqw384Oq>`_

Notes
-----
This implementation is inspired by the official one https://github.com/thuml/Time-Series-Library

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import SegRNN

__all__ = [
    "SegRNN",
]
