"""
Constant learning rate scheduler.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .base import LRScheduler, logger


class ConstantLR(LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches
    a pre-defined milestone: total_iters. Notice that such decay can happen simultaneously with other changes
    to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    factor: float, default=1./3.
        The number we multiply learning rate until the milestone.

    total_iters: int, default=5,
        The number of steps that the scheduler decays the learning rate.

    last_epoch: int
        The index of last epoch. Default: -1.

    verbose: bool
        If ``True``, prints a message to stdout for each update. Default: ``False``.

    Notes
    -----
    This class works the same with ``torch.optim.lr_scheduler.ConstantLR``.
    The only difference that is also why we implement them is that you don't have to pass according optimizers
    into them immediately while initializing them.

    Example
    -------
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.025   if epoch == 0
    >>> # lr = 0.025   if epoch == 1
    >>> # lr = 0.025   if epoch == 2
    >>> # lr = 0.025   if epoch == 3
    >>> # lr = 0.05    if epoch >= 4
    >>> # xdoctest: +SKIP
    >>> scheduler = ConstantLR(factor=0.5, total_iters=4)
    >>> adam = pypots.optim.Adam(lr=1e-3, lr_scheduler=scheduler)

    """

    def __init__(self, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False):
        super().__init__(last_epoch, verbose)
        if factor > 1.0 or factor < 0:
            raise ValueError("Constant multiplicative factor expected to be between 0 and 1.")

        self.factor = factor
        self.total_iters = total_iters

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning(
                "⚠️ To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
            )

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters or (self.last_epoch != self.total_iters):
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch == self.total_iters:
            return [group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            base_lr * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]
