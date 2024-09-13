"""
Multistep learning rate scheduler.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from bisect import bisect_right
from collections import Counter

from .base import LRScheduler, logger


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    Notice that such decay can happen simultaneously with other changes to the learning rate from outside this
    scheduler. When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    milestones: list,
        List of epoch indices. Must be increasing.

    gamma: float, default=0.1,
        Multiplicative factor of learning rate decay.

    last_epoch: int
        The index of last epoch. Default: -1.

    verbose: bool
        If ``True``, prints a message to stdout for each update. Default: ``False``.

    Notes
    -----
    This class works the same with ``torch.optim.lr_scheduler.MultiStepLR``.
    The only difference that is also why we implement them is that you don't have to pass according optimizers
    into them immediately while initializing them.

    Example
    -------
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 80
    >>> # lr = 0.0005   if epoch >= 80
    >>> # xdoctest: +SKIP
    >>> scheduler = MultiStepLR(milestones=[30,80], gamma=0.1)
    >>> adam = pypots.optim.Adam(lr=1e-3, lr_scheduler=scheduler)

    """

    def __init__(self, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(last_epoch, verbose)
        self.milestones = Counter(milestones)
        self.gamma = gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning(
                "⚠️ To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
            )

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch) for base_lr in self.base_lrs]
