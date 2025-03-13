"""
The implementation of GPT4TS for the partially-observed time-series forecasting task.

Refer to the paper
`Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin.
One Fits All: Power General Time Series Analysis by Pretrained LM.
NeurIPS 2023.
<https://openreview.net/forum?id=gMS6FVZvmF>`_

Notes
-----
This implementation is inspired by the official one https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import GPT4TS

__all__ = [
    "GPT4TS",
]
