"""
The package including the modules of ModernTCN.

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


from .backbone import BackboneModernTCN

__all__ = [
    "BackboneModernTCN",
]
