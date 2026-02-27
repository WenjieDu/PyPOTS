"""
The package of the partially-observed time-series classification model SeFT.

Refer to the paper
`Max Horn, Michael Moor, Christian Bock, Bastian Rieck, and Karsten Borgwardt.
Set Functions for Time Series.
In the 37th International Conference on Machine Learning, 2020.
<https://proceedings.mlr.press/v119/horn20a.html>`_

Notes
-----
This implementation is inspired by the official one https://github.com/BorgwardtLab/Set_Functions_for_Time_Series

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import SeFT

__all__ = [
    "SeFT",
]
