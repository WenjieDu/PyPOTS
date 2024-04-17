"""
The package including the modules of Raindrop.

Refer to the paper
`Xiang Zhang, Marko Zeman, Theodoros Tsiligkaridis, and Marinka Zitnik.
Graph-guided network for irregularly sampled multivariate time series.
In ICLR, 2022.
<https://openreview.net/forum?id=Kwm8I7dU-l5>`_

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneRaindrop

__all__ = [
    "BackboneRaindrop",
]
