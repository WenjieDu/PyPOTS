"""
The package of the partially-observed time-series anomaly detectoin model FEDformer.

Refer to the paper
`Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin.
FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting.
In ICML, volume 162 of Proceedings of Machine Learning Research, pages 27268–27286. PMLR, 17–23 Jul 2022.
<https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/MAZiqing/FEDformer

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause


from .model import FEDformer

__all__ = [
    "FEDformer",
]
