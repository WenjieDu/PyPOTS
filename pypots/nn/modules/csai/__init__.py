"""
The package including the modules of CSAI.

Refer to the paper
`Linglong Qian, Zina Ibrahim, Hugh Logan Ellis, Ao Zhang, Yuezhou Zhang, Tao Wang, Richard Dobson.
Knowledge Enhanced Conditional Imputation for Healthcare Time-series.
In Arxiv, 2024.
<https://arxiv.org/abs/2312.16713>`_

Notes
-----
This implementation is inspired by the official one the official implementation https://github.com/LinglongQian/CSAI.

"""

# Created by Joseph Arul Raj <joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

from .backbone import BackboneCSAI, BackboneBCSAI
from .layers import FeatureRegression, Decay, Decay_obs, PositionalEncoding, Conv1dWithInit, TorchTransformerEncoder

__all__ = [
    "BackboneCSAI",
    "BackboneBCSAI",
    "FeatureRegression",
    "Decay",
    "Decay_obs",
    "PositionalEncoding",
    "Conv1dWithInit",
    "TorchTransformerEncoder",
]
