"""
The package of the partially-observed time-series imputation model SCINet.

Refer to the paper
`Minhao LIU, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia LAI, Lingna Ma, and Qiang Xu.
"SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction".
In Advances in Neural Information Processing Systems, 2022.
<https://proceedings.neurips.cc/paper_files/paper/2022/file/266983d0949aed78a16fa4782237dea7-Paper-Conference.pdf>`_


Notes
-----
This implementation is inspired by the official one https://github.com/cure-lab/SCINet

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import SCINet

__all__ = [
    "SCINet",
]
