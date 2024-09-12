"""
Exponential learning rate scheduler.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .base import LRScheduler, logger


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    gamma: float,
        Multiplicative factor of learning rate decay.

    last_epoch: int
        The index of last epoch. Default: -1.

    verbose: bool
        If ``True``, prints a message to stdout for each update. Default: ``False``.

    Notes
    -----
    This class works the same with ``torch.optim.lr_scheduler.ExponentialLR``.
    The only difference that is also why we implement them is that you don't have to pass according optimizers
    into them immediately while initializing them.

    Example
    -------
    >>> scheduler = ExponentialLR(gamma=0.1)
    >>> adam = pypots.optim.Adam(lr=1e-3, lr_scheduler=scheduler)

    """

    def __init__(self, gamma, last_epoch=-1, verbose=False):
        super().__init__(last_epoch, verbose)

        self.gamma = gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning(
                "⚠️ To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]
