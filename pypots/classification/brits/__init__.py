"""
The package of the partially-observed time-series classification model BRITS.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import BRITS

__all__ = [
    "BRITS",
]
