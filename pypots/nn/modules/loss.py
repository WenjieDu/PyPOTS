"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .metric import BaseMetric
from ..functional import calc_mse


class BaseLoss(BaseMetric):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, prediction, target):
        raise NotImplementedError


class MAE_Loss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        return calc_mse(prediction, target, mask)
