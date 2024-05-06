"""
The package including the modules of PatchTST.

Refer to the paper
`Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.
A time series is worth 64 words: Long-term forecasting with transformers.
In ICLR, 2023.
<https://openreview.net/pdf?id=Jbdc0vTOcol>`_

Notes
-----
This implementation is inspired by the official one https://github.com/yuqinie98/PatchTST

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .autoencoder import PatchtstEncoder
from .layers import PatchEmbedding, RegressionHead, ClassificationHead, PredictionHead

__all__ = [
    "PatchtstEncoder",
    "PatchEmbedding",
    "RegressionHead",
    "ClassificationHead",
    "PredictionHead",
]
