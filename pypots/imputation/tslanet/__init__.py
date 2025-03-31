"""
The package of the partially-observed time-series imputation model TSLANet.

Refer to the paper
`Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, and Xiaoli Li.
TSLANet: Rethinking Transformers for Time Series Representation Learning.
ICML 2024.
<https://raw.githubusercontent.com/mlresearch/v235/main/assets/eldele24a/eldele24a.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/emadeldeen24/TSLANet

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import TSLANet

__all__ = [
    "TSLANet",
]
