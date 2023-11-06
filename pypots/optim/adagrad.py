"""
The optimizer wrapper for PyTorch Adagrad.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Iterable, Optional

from torch.optim import Adagrad as torch_Adagrad

from .base import Optimizer
from .lr_scheduler.base import LRScheduler


class Adagrad(Optimizer):
    """The optimizer wrapper for PyTorch Adagrad :class:`torch.optim.Adagrad`.

    Parameters
    ----------
    lr : float
        The learning rate of the optimizer.

    lr_decay : float
        Learning rate decay.

    weight_decay : float
        Weight decay (L2 penalty).

    eps : float
        Term added to the denominator to improve numerical stability.

    initial_accumulator_value : float
        A floating point value. Starting value for the accumulators, must be positive.

    lr_scheduler : pypots.optim.lr_scheduler.base.LRScheduler
        The learning rate scheduler of the optimizer.

    """

    def __init__(
        self,
        lr: float = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0.01,
        initial_accumulator_value: float = 0.01,  # it is set as 0 in the torch implementation, but delta shouldn't be 0
        eps: float = 1e-08,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr, lr_scheduler)
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params :
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.

        """
        self.torch_optimizer = torch_Adagrad(
            params=params,
            lr=self.lr,
            lr_decay=self.lr_decay,
            weight_decay=self.weight_decay,
            initial_accumulator_value=self.initial_accumulator_value,
            eps=self.eps,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.init_scheduler(self.torch_optimizer)
