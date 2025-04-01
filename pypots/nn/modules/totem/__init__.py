"""
The package including the modules of TOTEM.

Refer to the paper
`Sabera J Talukder, Yisong Yue, and Georgia Gkioxari.
TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis.
In TMLR, 2024.
<https://openreview.net/forum?id=QlTLkH6xRC>`_

Notes
-----
This implementation is inspired by the official one https://github.com/SaberaTalukder/TOTEM

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .autoencoder import VQVAE

__all__ = [
    "VQVAE",
]
