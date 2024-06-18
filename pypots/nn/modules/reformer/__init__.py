"""
The package including the modules of Reformer.

Refer to the paper
`Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya.
"Reformer: The Efficient Transformer".
In International Conference on Learning Representations, 2020.
<https://openreview.net/pdf?id=rkgNKkHtvB>`_

Notes
-----
This implementation is inspired by the official one https://github.com/google/trax/tree/master/trax/models/reformer and
https://github.com/lucidrains/reformer-pytorch

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .autoencoder import ReformerEncoder

__all__ = [
    "ReformerEncoder",
]
