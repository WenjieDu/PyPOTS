"""
The package including the modules of FITS.

Refer to the paper
`Zhijian Xu, Ailing Zeng, and Qiang Xu.
FITS: Modeling Time Series with 10k parameters.
In The Twelfth International Conference on Learning Representations, 2024.
<https://openreview.net/pdf?id=bWcnvZ3qMb>`_

Notes
-----
This implementation is inspired by the official one https://github.com/VEWOXIC/FITS

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneFITS


__all__ = [
    "BackboneFITS",
]
