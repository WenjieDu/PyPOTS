"""
The package of the partially-observed time-series imputation model Informer.

Refer to the paper "Wu, H., Xu, J., Wang, J., & Long, M. (2021).
Informer: Decomposition transformers with auto-correlation for long-term series forecasting. NeurIPS 2021.".

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import Informer

__all__ = [
    "Informer",
]
