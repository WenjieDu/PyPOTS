"""
The package of the partially-observed time-series imputation model Imputeformer.

Refer to the papers
`Tong Nie, Guoyang Qin, Wei Ma, Yuewen Mei, Jian Sun.
"ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation"
KDD 2024.
<https://doi.org/10.48550/arXiv.2312.01728>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import Imputeformer

__all__ = [
    "Imputeformer",
]
