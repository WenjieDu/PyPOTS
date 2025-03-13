"""
The package of the partially-observed time-series imputation model SegRNN.

Refer to the paper
`Lin, Shengsheng and Lin, Weiwei and Wu, Wentai and Zhao, Feiyu and Mo, Ruichao and Zhang, Haotong.
Segrnn: Segment recurrent neural network for long-term time series forecasting.
arXiv preprint arXiv:2308.11200.
<https://arxiv.org/abs/2308.11200>`_

Notes
-----
This implementation is inspired by the official one https://github.com/lss-1138/SegRNN

"""

# Created by Shengsheng Lin


from .model import SegRNN

__all__ = [
    "SegRNN",
]
