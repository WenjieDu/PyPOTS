"""
The package of the forecasting model TEFN.

Refer to the paper
`Tianxiang Zhan, Yuanpeng He, Yong Deng, and Zhen Li.
Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting.
In Arxiv, 2024.
<https://arxiv.org/abs/2405.06419>`_

Notes
-----
This implementation is transfered from the official one https://github.com/ztxtech/Time-Evidence-Fusion-Network

"""

# Created by Tianxiang Zhan <zhantianxianguestc@hotmail.com>
# License: BSD-3-Clause


from .backbone import BackboneTEFN

__all__ = [
    "BackboneTEFN",
]
