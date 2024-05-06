"""
The package including the modules of Pyraformer.

Refer to the paper
`Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X. Liu, and Schahram Dustdar.
"Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting".
International Conference on Learning Representations. 2022.
<https://openreview.net/pdf?id=0EXmFzUn5I>`_

Notes
-----
This implementation is inspired by the official one https://github.com/ant-research/Pyraformer

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .autoencoder import PyraformerEncoder

__all__ = [
    "PyraformerEncoder",
]
