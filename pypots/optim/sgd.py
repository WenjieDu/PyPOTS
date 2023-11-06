"""
The optimizer wrapper for PyTorch SGD :class:`torch.optim.SGD`.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Iterable, Optional

from torch.optim import SGD as torch_SGD

from .base import Optimizer
from .lr_scheduler.base import LRScheduler


class SGD(Optimizer):
    """The optimizer wrapper for PyTorch SGD :class:`torch.optim.SGD`.

    Parameters
    ----------
    lr : float
        The learning rate of the optimizer.

    momentum : float
        Momentum factor.

    weight_decay : float
        Weight decay (L2 penalty).

    dampening : float
        Dampening for momentum.

    nesterov : bool
        Whether to enable Nesterov momentum.

    lr_scheduler : pypots.optim.lr_scheduler.base.LRScheduler
        The learning rate scheduler of the optimizer.

    """

    def __init__(
        self,
        lr: float = 0.001,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr, lr_scheduler)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params :
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.

        """
        self.torch_optimizer = torch_SGD(
            params=params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.init_scheduler(self.torch_optimizer)
