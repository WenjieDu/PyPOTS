"""
The package of the partially-observed time-series anomaly detection model iTransformer.

Refer to the papers
`Liu, Yong, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long.
"iTransformer: Inverted transformers are effective for time series forecasting."
ICLR 2024.
<https://openreview.net/pdf?id=JePfAI8fah>`_

Notes
-----
This implementation is inspired by the official one https://github.com/thuml/iTransformer

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from .model import iTransformer

__all__ = [
    "iTransformer",
]
