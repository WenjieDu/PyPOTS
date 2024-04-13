"""
The package including the modules of USGAN.

Refer to the paper
`Xiaoye Miao, Yangyang Wu, Jun Wang, Yunjun Gao, Xudong Mao, and Jianwei Yin.
Generative Semi-supervised Learning for Multivariate Time Series Imputation.
In AAAI, 35(10):8983–8991, May 2021.
<https://ojs.aaai.org/index.php/AAAI/article/view/17086/16893>`_

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneUSGAN

__all__ = [
    "BackboneUSGAN",
]
