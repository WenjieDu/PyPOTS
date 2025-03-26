"""
The implementation of TEFN for the partially-observed time-series anomaly detection task.

Refer to the paper
`Tianxiang Zhan, Yuanpeng He, Yong Deng, Zhen Li, Wenjie Du, and Qingsong Wen.
Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting.
In Arxiv, 2024.
<https://arxiv.org/abs/2405.06419>`_

Notes
-----
This implementation is transferred from the official one https://github.com/ztxtech/Time-Evidence-Fusion-Network

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import TEFN

__all__ = [
    "TEFN",
]
