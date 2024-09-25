"""
The package of the partially-observed time-series imputation model RevIN_SCINet.

Refer to the paper
`Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo.
"Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift".
In International Conference on Learning Representations, 2022.
<https://openreview.net/pdf?id=cGDAkQo1C0p>`_

Notes
-----
This implementation is inspired by the official one https://github.com/ts-kim/RevIN

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import RevIN_SCINet

__all__ = [
    "RevIN_SCINet",
]
