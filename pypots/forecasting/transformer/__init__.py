"""
The implementation of Transformer for the partially-observed time-series forecasting task.

Refer to the papers
`Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser,
and Illia Polosukhin.
Attention is all you need.
In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
<https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_

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
