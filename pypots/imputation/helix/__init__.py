"""
The package of the partially-observed time-series imputation model HELIX.

Refer to the paper
`MiBah Cat.
HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis 
for Time Series Imputation.
<paper_url_placeholder>`_

Notes
-----
HELIX employs rotary positional encoding for temporal dimension and learnable 
identity embeddings for feature dimension, combined with parallel and serial 
cross-dimensional attention mechanism.

"""

# Created by MiBah Cat <milaogou@gmail.com>
# License: BSD-3-Clause

from .model import HELIX

__all__ = [
    "HELIX",
]