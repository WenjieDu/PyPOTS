"""
The package including the modules of CRLI.

Refer to the paper
`Qianli Ma, Chuxin Chen, Sen Li, and Garrison W. Cottrell.
Learning Representations for Incomplete Time Series Clustering.
In AAAI, 35(10):8837–8846, May 2021.
<https://ojs.aaai.org/index.php/AAAI/article/view/17070>`_

Notes
-----
This implementation is inspired by the official one https://github.com/qianlima-lab/CRLI

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneCRLI
from .layers import CrliGenerator, CrliDecoder, CrliDiscriminator

__all__ = [
    "BackboneCRLI",
    "CrliGenerator",
    "CrliDecoder",
    "CrliDiscriminator",
]
