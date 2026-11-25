"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.functional.error import (
    calc_mae,
    calc_mre,
    calc_mse,
    calc_quantile_crps,
    calc_quantile_crps_sum,
    calc_rmse,
)
from ..logging import logger

# pypots.nn.functional.error is deprecated, and moved to pypots.nn.functional.error
logger.warning("🚨 Please import from pypots.nn.functional.error instead of pypots.nn.functional.error")

__all__ = [
    "calc_mae",
    "calc_mre",
    "calc_mse",
    "calc_quantile_crps",
    "calc_quantile_crps_sum",
    "calc_rmse",
]
