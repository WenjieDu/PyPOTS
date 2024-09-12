"""
Multiplicative learning rate scheduler.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .base import LRScheduler, logger


class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function.
    When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    lr_lambda: Callable or list,
        A function which computes a multiplicative factor given an integer parameter epoch, or a list of such
        functions, one for each group in optimizer.param_groups.

    last_epoch: int,
        The index of last epoch. Default: -1.

    verbose: bool,
        If ``True``, prints a message to stdout for each update. Default: ``False``.

    Notes
    -----
    This class works the same with ``torch.optim.lr_scheduler.MultiplicativeLR``.
    The only difference that is also why we implement them is that you don't have to pass according optimizers
    into them immediately while initializing them.

    Example
    -------
    >>> lmbda = lambda epoch: 0.95
    >>> # xdoctest: +SKIP
    >>> scheduler = MultiplicativeLR(lr_lambda=lmbda)
    >>> adam = pypots.optim.Adam(lr=1e-3, lr_scheduler=scheduler)

    """

    def __init__(self, lr_lambda, last_epoch=-1, verbose=False):
        super().__init__(last_epoch, verbose)
        self.lr_lambda = lr_lambda
        self.lr_lambdas = None

    def init_scheduler(self, optimizer):
        if not isinstance(self.lr_lambda, list) and not isinstance(self.lr_lambda, tuple):
            self.lr_lambdas = [self.lr_lambda] * len(optimizer.param_groups)
        else:
            if len(self.lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} lr_lambdas, but got {}".format(len(optimizer.param_groups), len(self.lr_lambda))
                )
            self.lr_lambdas = list(self.lr_lambda)

        super().init_scheduler(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning(
                "⚠️ To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
            )

        if self.last_epoch > 0:
            return [
                group["lr"] * lmbda(self.last_epoch)
                for lmbda, group in zip(self.lr_lambdas, self.optimizer.param_groups)
            ]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]
