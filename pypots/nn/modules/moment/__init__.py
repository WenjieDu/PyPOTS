"""
The package including the modules of MOMENT.

Refer to the paper
`Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski.
"MOMENT: A Family of Open Time-series Foundation Models".
In ICML, 2024.
<https://proceedings.mlr.press/v235/goswami24a.html>`_

Notes
-----
This implementation is inspired by the official one
https://github.com/moment-timeseries-foundation-model/moment-research

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneMOMENT, SUPPORTED_HUGGINGFACE_MODELS

__all__ = [
    "BackboneMOMENT",
    "SUPPORTED_HUGGINGFACE_MODELS",
]
