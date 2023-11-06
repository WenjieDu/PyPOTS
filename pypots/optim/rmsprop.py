"""
The optimizer wrapper for PyTorch RMSprop.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Iterable, Optional

from torch.optim import RMSprop as torch_RMSprop

from .base import Optimizer
from .lr_scheduler.base import LRScheduler


class RMSprop(Optimizer):
    """The optimizer wrapper for PyTorch RMSprop :class:`torch.optim.RMSprop`.

    Parameters
    ----------
    lr : float
        The learning rate of the optimizer.

    momentum : float
        Momentum factor.

    alpha : float
        Smoothing constant.

    eps : float
        Term added to the denominator to improve numerical stability.

    centered : bool
        If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

    weight_decay : float
        Weight decay (L2 penalty).

    lr_scheduler : pypots.optim.lr_scheduler.base.LRScheduler
        The learning rate scheduler of the optimizer.

    """

    def __init__(
        self,
        lr: float = 0.001,
        momentum: float = 0,
        alpha: float = 0.99,
        eps: float = 1e-08,
        centered: bool = False,
        weight_decay: float = 0,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr, lr_scheduler)
        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        self.centered = centered
        self.weight_decay = weight_decay

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params :
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.

        """
        self.torch_optimizer = torch_RMSprop(
            params=params,
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.alpha,
            eps=self.eps,
            centered=self.centered,
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.init_scheduler(self.torch_optimizer)
