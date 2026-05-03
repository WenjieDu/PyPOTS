"""
The package of the partially-observed time-series imputation model HELIX.

Refer to the paper
`Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, and Shen Qu.
HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation.
ICML (spotlight), 2026.
<https://openreview.net/forum?id=FN20iuPnEU>`_

Notes
-----
Refer to the repo https://github.com/milaogou/HELIX for details.

"""

# Created by Fengming Zhang <milaogou@gmail.com>
# License: BSD-3-Clause

from .model import HELIX

__all__ = [
    "HELIX",
]
