"""
The package of the partially-observed time-series anomaly detection model ETSformer.

Refer to the paper
`Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi.
ETSformer: Exponential smoothing transformers for time-series forecasting.
In ICLR, 2023.
<https://openreview.net/pdf?id=5m_3whfo483>`_

Notes
-----
This implementation is inspired by the official one https://github.com/salesforce/ETSformer

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause


from .model import ETSformer

__all__ = [
    "ETSformer",
]
