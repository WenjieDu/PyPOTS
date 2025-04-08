"""
The package of the partially-observed time-series forecasting model ModernTCN.

Refer to the paper
`Donghao Luo, and Xue Wang.
ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis.
In The Twelfth International Conference on Learning Representations. 2024.
<https://openreview.net/pdf?id=vpJMJerXHU>`_

Notes
-----
This implementation is inspired by the official one https://github.com/luodhhh/ModernTCN

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import ModernTCN

__all__ = [
    "ModernTCN",
]
