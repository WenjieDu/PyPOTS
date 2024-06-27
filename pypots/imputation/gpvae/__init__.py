"""
The package of the partially-observed time-series imputation model GP-VAE.

Refer to the paper
`Vincent Fortuin, Dmitry Baranchuk, Gunnar Rätsch, and Stephan Mandt.
GP-VAE: Deep probabilistic time series imputation.
In International conference on artificial intelligence and statistics, pages 1651–1661. PMLR, 2020.
<http://proceedings.mlr.press/v108/fortuin20a/fortuin20a.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/ratschlab/GP-VAE

"""

# Created by Jun Wang <jwangfx@connect.ust.hk>
# License: BSD-3-Clause

from .model import GPVAE

__all__ = [
    "GPVAE",
]
