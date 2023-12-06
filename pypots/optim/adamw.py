"""
The optimizer wrapper for PyTorch AdamW.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Iterable, Tuple, Optional

from torch.optim import AdamW as torch_AdamW

from .base import Optimizer
from .lr_scheduler.base import LRScheduler


class AdamW(Optimizer):
    """The optimizer wrapper for PyTorch AdamW :class:`torch.optim.AdamW`.

    Parameters
    ----------
    lr : float
        The learning rate of the optimizer.

    betas : Tuple[float, float]
        Coefficients used for computing running averages of gradient and its square.

    eps : float
        Term added to the denominator to improve numerical stability.

    weight_decay : float
        Weight decay (L2 penalty).

    amsgrad : bool
        Whether to use the AMSGrad variant of this algorithm from the paper :cite:`reddi2018OnTheConvergence`.

    lr_scheduler : pypots.optim.lr_scheduler.base.LRScheduler
        The learning rate scheduler of the optimizer.

    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr, lr_scheduler)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params :
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.

        """
        self.torch_optimizer = torch_AdamW(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.init_scheduler(self.torch_optimizer)
