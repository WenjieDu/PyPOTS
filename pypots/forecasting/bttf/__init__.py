"""
The package of the partially-observed time-series forecasting model BTTF.

Refer to the paper "Chen, X., & Sun, L. (2021).
Bayesian Temporal Factorization for Multidimensional Time Series Prediction.
IEEE transactions on pattern analysis and machine intelligence."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from .model import BTTF

__all__ = [
    "BTTF",
]
