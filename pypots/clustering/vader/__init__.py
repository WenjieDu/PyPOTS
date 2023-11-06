"""
The package of the partially-observed time-series clustering model VaDER.

Refer to the paper "Jong, J.D., Emon, M.A., Wu, P., Karki, R., Sood, M., Godard, P., Ahmad, A., Vrooman, H.A.,
Hofmann-Apitius, M., & Fröhlich, H. (2019).
Deep learning for clustering of multivariate clinical patient trajectories with missing values. GigaScience."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import VaDER

__all__ = [
    "VaDER",
]
