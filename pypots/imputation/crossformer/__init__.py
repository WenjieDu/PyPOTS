"""
The package of the partially-observed time-series imputation model Transformer.

Refer to the paper
`Yunhao Zhang and Junchi Yan.
Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting.
In The 11th ICLR, 2023.
<https://openreview.net/pdf?id=vSVLM2j9eie>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import Crossformer

__all__ = [
    "Crossformer",
]
