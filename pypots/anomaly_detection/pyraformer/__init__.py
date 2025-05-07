"""
The package of the partially-observed time-series anomaly detection model Pyraformer.

Refer to the paper
`Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X. Liu, and Schahram Dustdar.
"Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting".
International Conference on Learning Representations. 2022.
<https://openreview.net/pdf?id=0EXmFzUn5I>`_

Notes
-----
This implementation is inspired by the official one https://github.com/ant-research/Pyraformer

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause


from .model import Pyraformer

__all__ = [
    "Pyraformer",
]
