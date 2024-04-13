"""
The package including the modules of GRU-D.

Refer to the paper "Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018).
Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports."
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneGRUD
from .layers import TemporalDecay

__all__ = [
    "BackboneGRUD",
    "TemporalDecay",
]
