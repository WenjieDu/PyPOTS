"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
from torch.nn.modules.loss import _Loss

from ..functional import (
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mre,
)


class Criterion(_Loss):
    def __init__(
        self,
        lower_better: bool = True,
    ):
        """The base class for all class implementation loss functions and metrics in PyPOTS.

        Parameters
        ----------
        lower_better :
            Whether the lower value of the criterion directs to a better model performance.
            Default as True which is the case for most loss functions (e.g. MSE, Cross Entropy).
            If False, it makes that the higher value leads to a better model performance (e.g. Accuracy).

        """
        super().__init__()
        self.lower_better = lower_better

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """The criterion calculation process.

        Parameters
        ----------
        logits:
            The model outputs, predicted unnormalized logits.

        targets:
            The ground truth values.

        """
        raise NotImplementedError


class MSE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_mse(logits, targets, masks)
        return value


class MAE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_mae(logits, targets, masks)
        return value


class RMSE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_rmse(logits, targets, masks)
        return value


class MRE(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        value = calc_mre(logits, targets, masks)
        return value


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.nn.functional.cross_entropy(logits, targets)
        return value


class NLL(Criterion):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.nn.functional.nll_loss(log_probs, targets)
        return value
