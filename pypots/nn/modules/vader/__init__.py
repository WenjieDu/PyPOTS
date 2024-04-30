"""
The package including the modules of VaDER.

Refer to the paper
`Johann de Jong, Mohammad Asif Emon, Ping Wu, Reagon Karki, Meemansa Sood, Patrice Godard, Ashar Ahmad, Henri Vrooman,
Martin Hofmann-Apitius, and Holger Fröhlich.
Deep learning for clustering of multivariate clinical patient trajectories with missing values.
GigaScience, 8(11):giz134, November 2019.
<https://academic.oup.com/gigascience/article-pdf/8/11/giz134/30797160/giz134.pdf>`_

Notes
-----
This implementation is inspired by the official one https://github.com/johanndejong/VaDER

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneVaDER
from .layers import PeepholeLSTMCell, ImplicitImputation, GMMLayer

__all__ = [
    "BackboneVaDER",
    "PeepholeLSTMCell",
    "ImplicitImputation",
    "GMMLayer",
]
