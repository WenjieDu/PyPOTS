"""
The package of the partially-observed time-series classification model Raindrop.

Refer to the paper "Zhang, X., Zeman, M., Tsiligkaridis, T., & Zitnik, M. (2022).
Graph-Guided Network for Irregularly Sampled Multivariate Time Series. ICLR 2022."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import Raindrop

__all__ = [
    "Raindrop",
]
