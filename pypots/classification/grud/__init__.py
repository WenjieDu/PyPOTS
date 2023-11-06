"""
The package of the partially-observed time-series classification model GRU-D.

Refer to the paper "Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018).
Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import GRUD

__all__ = [
    "GRUD",
]
