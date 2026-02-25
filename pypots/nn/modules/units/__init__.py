"""
The package including the modules of UniTS.

Refer to the paper
`Shanghua Gao, Teddy Koker, Owen Queen, Thomas Hartvigsen, Theodoros Tsiligkaridis, Marinka Zitnik.
UniTS: Building a Unified Time Series Model.
In NeurIPS, 2024.
<https://arxiv.org/pdf/2403.00131.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/mims-harvard/UniTS

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .layers import (
    PatchEmbedding,
    LearnablePositionalEmbedding,
    DynamicLinear,
    BasicBlock,
    ForecastHead,
)

__all__ = [
    "PatchEmbedding",
    "LearnablePositionalEmbedding",
    "DynamicLinear",
    "BasicBlock",
    "ForecastHead",
]
