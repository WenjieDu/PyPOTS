"""
The package of the partially-observed time-series imputation model PITS.

Refer to the paper
`Seunghan Lee, Taeyoung Park, and Kibok Lee.
Learning to Embed Time Series Patches Independently.
In ICLR, 2024.
<https://openreview.net/forum?id=WS7GuBDFa2>`_

Notes
-----
This implementation is inspired by the official one https://github.com/seunghan96/pits

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import PITS

__all__ = [
    "PITS",
]
