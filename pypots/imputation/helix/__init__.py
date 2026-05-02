"""
The package of the partially-observed time-series imputation model HELIX.

Refer to the paper
`Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, and Shen Qu.
HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation.
ICML (spotlight), 2026.
<>`_

Notes
-----
HELIX employs rotary positional encoding for temporal dimension and learnable
identity embeddings for feature dimension, combined with parallel and serial
cross-dimensional attention mechanism.

This implementation is inspired by the official one https://github.com/milaogou/HELIX

"""

# Created by Fengming Zhang <milaogou@gmail.com>
# License: BSD-3-Clause

from .model import HELIX

__all__ = [
    "HELIX",
]
