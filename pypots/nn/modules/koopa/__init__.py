"""
The package including the modules of Koopa.

Refer to the paper
`Yong Liu, Chenyu Li, Jianmin Wang, and Mingsheng Long.
"Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors".
Advances in Neural Information Processing Systems 36 (2023).
<https://proceedings.neurips.cc/paper_files/paper/2023/file/28b3dc0970fa4624a63278a4268de997-Paper-Conference.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/thuml/Koopa

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneKoopa
from .layers import FourierFilter, MLP, TimeVarKP, TimeInvKP

__all__ = [
    "BackboneKoopa",
    "MLP",
    "FourierFilter",
    "TimeVarKP",
    "TimeInvKP",
]
