"""
The package of the partially-observed time-series imputation model TKAN
(Temporal Kolmogorov-Arnold Networks).

Refer to the paper
`Remi Genet and Hugo Inzirillo.
"TKAN: Temporal Kolmogorov-Arnold Networks".
arXiv preprint arXiv:2405.07344, 2024.
<https://arxiv.org/abs/2405.07344>`_

Notes
-----
This implementation is inspired by the official one https://github.com/remigenet/TKAN

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import TKAN

__all__ = [
    "TKAN",
]
