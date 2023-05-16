"""
The package of the partially-observed time-series classification model GRUD.

Refer to the paper "Che, Z., Purushotham, S., Cho, K., Sontag, D.A., & Liu, Y. (2018).
Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from pypots.classification.grud.model import GRUD

__all__ = [
    "GRUD",
]
