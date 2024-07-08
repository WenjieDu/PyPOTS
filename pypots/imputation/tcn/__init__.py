"""
The package of the partially-observed time-series imputation model TCN.

Refer to the paper
`Shaojie Bai, J. Zico Kolter, and Vladlen Koltun.
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling".
arXiv preprint arXiv:1803.01271.
<https://arxiv.org/pdf/1803.01271>`_

Notes
-----
This implementation is inspired by the official one https://github.com/locuslab/TCN

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import TCN

__all__ = [
    "TCN",
]
