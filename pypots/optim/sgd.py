"""
The optimizer wrapper for PyTorch SGD.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Iterable

from torch.optim import SGD as torch_SGD

from pypots.optim.base import Optimizer


class SGD(Optimizer):
    """The optimizer wrapper for PyTorch SGD.
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

    Parameters
    ----------
    lr : float, default=0.001
        The learning rate of the optimizer.

    momentum : float, default=0
        Momentum factor.

    weight_decay : float, default=0.01
        Weight decay (L2 penalty).

    dampening : float, default = 0
        Dampening for momentum.

    nesterov : bool, default=False
        Whether to enable Nesterov momentum.

    """

    def __init__(
        self,
        lr: float = 0.001,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
    ):
        super().__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params : Iterable,
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
