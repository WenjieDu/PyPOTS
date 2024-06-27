"""
The package including the modules of M-RNN.

Refer to the paper
`Jinsung Yoon, William R. Zame, and Mihaela van der Schaar.
Estimating missing data in temporal data streams using multi-directional recurrent neural networks.
IEEE Transactions on Biomedical Engineering, 66(5):14771490, 2019.
<https://arxiv.org/pdf/1711.08742>`_

Notes
-----
This implementation is inspired by the official one
https://github.com/jsyoon0823/MRNN and https://github.com/WenjieDu/SAITS

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneMRNN
from .layers import MrnnFcnRegression


__all__ = [
    "BackboneMRNN",
    "MrnnFcnRegression",
]
