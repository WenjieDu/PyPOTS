"""
The package including the modules of ImputeFormer.

Refer to the paper
`Tong Nie, Guoyang Qin, Wei Ma, Yuewen Mei, Jian Sun.
ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation.
KDD, 2024.
<https://doi.org/10.48550/arXiv.2312.01728>`_

Notes
-----
This implementation is inspired by the official one https://github.com/tongnie/ImputeFormer

"""

# Created by Tong Nie <nietong@tongji.edu.cn> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .attention import EmbeddedAttentionLayer, ProjectedAttentionLayer
from .mlp import MLP

__all__ = [
    "EmbeddedAttentionLayer",
    "ProjectedAttentionLayer",
    "MLP",
]
