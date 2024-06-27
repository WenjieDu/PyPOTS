"""
The package including the modules of DLinear.

Refer to the paper
`Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu.
Are transformers effective for time series forecasting?
In AAAI, volume 37, pages 11121–11128, Jun. 2023.
<https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089>`_

Notes
-----
This implementation is inspired by the official one https://github.com/cure-lab/LTSF-Linear

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .backbone import BackboneDLinear

__all__ = [
    "BackboneDLinear",
]
