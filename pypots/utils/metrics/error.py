"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ..logging import logger
from ...nn.functional.error import (
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mre,
    calc_quantile_crps,
    calc_quantile_crps_sum,
)

# pypots.nn.functional.error is deprecated, and moved to pypots.nn.functional.error
logger.warning("🚨 Please import from pypots.nn.functional.error instead of pypots.nn.functional.error")

__all__ = [
    "calc_mae",
    "calc_mse",
    "calc_rmse",
    "calc_mre",
    "calc_quantile_crps",
    "calc_quantile_crps_sum",
]
