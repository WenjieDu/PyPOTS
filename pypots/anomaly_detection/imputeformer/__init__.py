"""
The package of the partially-observed time-series anomaly detection model ImputeFormer.

Refer to the papers
`Tong Nie, Guoyang Qin, Wei Ma, Yuewen Mei, Jian Sun.
"ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation"
KDD 2024.
<https://doi.org/10.48550/arXiv.2312.01728>`_

Notes
-----
This implementation is inspired by the official one https://github.com/tongnie/ImputeFormer

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import ImputeFormer

__all__ = [
    "ImputeFormer",
]
