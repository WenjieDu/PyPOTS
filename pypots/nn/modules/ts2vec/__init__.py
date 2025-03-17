"""
The package including the modules of TS2Vec.

Refer to the paper
`Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, Bixiong Xu.
"TS2Vec: Towards Universal Representation of Time Series".
In AAAI 2022.
<https://ojs.aaai.org/index.php/AAAI/article/view/20881/20640>`_

Notes
-----
This implementation is inspired by the official one https://github.com/zhihanyue/ts2vec

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import TS2VecEncoder

__all__ = [
    "TS2VecEncoder",
]
