"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch

from .metric import BaseMetric
from ..functional import (
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mre,
    calc_quantile_crps,
    calc_quantile_crps_sum,
)


class BaseLoss(BaseMetric):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, prediction, target):
        raise NotImplementedError


class MSE(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        value = calc_mse(prediction, target, mask)
        return value


class MAE(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        value = calc_mae(prediction, target, mask)
        return value


class RMSE(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        value = calc_rmse(prediction, target, mask)
        return value


class MRE(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        value = calc_mre(prediction, target, mask)
        return value


class QuantileCRPS(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        value = calc_quantile_crps(prediction, target, mask)
        return value


class QuantileCRPS_Sum(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        value = calc_quantile_crps_sum(prediction, target, mask)
        return value


class CrossEntropy(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        value = torch.nn.functional.cross_entropy(prediction, target)
        return value
