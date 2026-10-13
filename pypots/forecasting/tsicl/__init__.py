"""
The package of the partially-observed time-series forecasting method TS-ICL.

Refer to the paper
`Etienne Le Naour, Tahar Nabil, and Adrien Petralia.
"TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning".
arXiv preprint arXiv:2606.05878, 2026.
<https://arxiv.org/abs/2606.05878>`_

Notes
-----
This implementation is ported from the official one https://github.com/EDF-Lab/ts-icl.
Official pretrained weights hosted on Hugging Face are subject to the TS-ICL model license.

"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

from .model import TSICL

__all__ = [
    "TSICL",
]
