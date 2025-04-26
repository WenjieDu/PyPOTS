"""
The package of the partially-observed time-series anomaly detection model Reformer.

Refer to the paper
`Kitaev, Nikita, Łukasz Kaiser, and Anselm Levskaya.
Reformer: The Efficient Transformer.
International Conference on Learning Representations, 2020.
<https://openreview.net/pdf?id=rkgNKkHtvB>`_

Notes
-----
This implementation is inspired by the official one https://github.com/google/trax/tree/master/trax/models/reformer and
https://github.com/lucidrains/reformer-pytorch

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import Reformer

__all__ = [
    "Reformer",
]
