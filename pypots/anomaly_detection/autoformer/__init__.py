"""
The package of the partially-observed time-series anomaly detection model Autoformer.

Refer to the paper
`Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long.
Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting.
In Advances in Neural Information Processing Systems, volume 34, pages 22419–22430. Curran Associates, Inc., 2021.
<https://proceedings.neurips.cc/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/thuml/Autoformer

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import Autoformer

__all__ = [
    "Autoformer",
]
