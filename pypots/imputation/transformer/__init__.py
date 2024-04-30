"""
The package of the partially-observed time-series imputation model Transformer.

Refer to the papers
`Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser,
and Illia Polosukhin.
Attention is all you need.
In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
<https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_
and
`Wenjie Du, David Cote, and Yan Liu.
SAITS: Self-Attention-based Imputation for Time Series.
Expert Systems with Applications, 219:119619, 2023.
<https://arxiv.org/pdf/2202.08516>`_

Notes
-----
This implementation is inspired by https://github.com/WenjieDu/SAITS
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import Transformer

__all__ = [
    "Transformer",
]
