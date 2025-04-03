"""
The package including the modules of Inception model.

Refer to the paper
`Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
Vincent Vanhoucke, Andrew Rabinovich.
Going deeper with convolutions.
CVPR 2015.
<https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf>`_


"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .layers import (
    InceptionBlockV1,
    InceptionTransBlockV1,
)

__all__ = [
    "InceptionBlockV1",
    "InceptionTransBlockV1",
]
