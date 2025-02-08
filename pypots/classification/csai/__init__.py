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

from .model import CSAI

__all__ = [
    "CSAI",
]
