"""
The optimizer wrapper for PyTorch Adadelta.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Iterable, Optional

from torch.optim import Adadelta as torch_Adadelta

from .base import Optimizer
from .lr_scheduler.base import LRScheduler


class Adadelta(Optimizer):
    """The optimizer wrapper for PyTorch Adadelta :class:`torch.optim.Adadelta`.

    Parameters
    ----------
    lr : float
        The learning rate of the optimizer.

    rho : float
        Coefficient used for computing a running average of squared gradients.

    eps : float
        Term added to the denominator to improve numerical stability.

    weight_decay : float
        Weight decay (L2 penalty).

    lr_scheduler : pypots.optim.lr_scheduler.base.LRScheduler
        The learning rate scheduler of the optimizer.

    """

    def __init__(
        self,
        lr: float = 0.01,
        rho: float = 0.9,
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr, lr_scheduler)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params :
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.

        """
        self.torch_optimizer = torch_Adadelta(
            params=params,
            lr=self.lr,
            rho=self.rho,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.init_scheduler(self.torch_optimizer)
