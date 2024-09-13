"""
Linear learning rate scheduler.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .base import LRScheduler, logger


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor until
    the number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can happen simultaneously
    with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    start_factor: float, default=1.0 / 3,
        The number we multiply learning rate in the first epoch. The multiplication factor changes towards
        end_factor in the following epochs.

    end_factor: float, default=1.0,
        The number we multiply learning rate at the end of linear changing process.

    total_iters: int, default=5,
        The number of iterations that multiplicative factor reaches to 1.

    last_epoch: int
        The index of last epoch. Default: -1.

    verbose: bool
        If ``True``, prints a message to stdout for each update. Default: ``False``.

    Notes
    -----
    This class works the same with ``torch.optim.lr_scheduler.LinearLR``.
    The only difference that is also why we implement them is that you don't have to pass according optimizers
    into them immediately while initializing them.

    Example
    -------
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.025    if epoch == 0
    >>> # lr = 0.03125  if epoch == 1
    >>> # lr = 0.0375   if epoch == 2
    >>> # lr = 0.04375  if epoch == 3
    >>> # lr = 0.05    if epoch >= 4
    >>> # xdoctest: +SKIP
    >>> scheduler = LinearLR(start_factor=0.5, total_iters=4)
    >>> adam = pypots.optim.Adam(lr=1e-3, lr_scheduler=scheduler)

    """

    def __init__(
        self,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(last_epoch, verbose)
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError("Starting multiplicative factor expected to be between 0 and 1.")

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning(
                "⚠️ To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
            )

        if self.last_epoch == 0:
            return [group["lr"] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor))
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters
            )
            for base_lr in self.base_lrs
        ]
