"""
Step learning rate scheduler.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .base import LRScheduler, logger


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    step_size: int,
        Period of learning rate decay.

    gamma: float, default=0.1,
        Multiplicative factor of learning rate decay.

    last_epoch: int
        The index of last epoch. Default: -1.

    verbose: bool
        If ``True``, prints a message to stdout for each update. Default: ``False``.

    Notes
    -----
    This class works the same with ``torch.optim.lr_scheduler.StepLR``.
    The only difference that is also why we implement them is that you don't have to pass according optimizers
    into them immediately while initializing them.

    Example
    -------
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 60
    >>> # lr = 0.0005   if 60 <= epoch < 90
    >>> # ...
    >>> # xdoctest: +SKIP
    >>> scheduler = StepLR(step_size=30, gamma=0.1)
    >>> adam = pypots.optim.Adam(lr=1e-3, lr_scheduler=scheduler)

    """

    def __init__(self, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(last_epoch, verbose)

        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning(
                "⚠️ To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
            )

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size) for base_lr in self.base_lrs]
