"""
The optimizer wrapper for PyTorch RMSprop.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Iterable

from torch.optim import RMSprop as torch_RMSprop

from .base import Optimizer


class RMSprop(Optimizer):
    """The optimizer wrapper for PyTorch RMSprop.
    https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop

    Parameters
    ----------
    lr :
        The learning rate of the optimizer.

    momentum :
        Momentum factor.

    alpha :
        Smoothing constant.

    eps :
        Term added to the denominator to improve numerical stability.

    centered :
        If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

    weight_decay :
        Weight decay (L2 penalty).

    """

    def __init__(
        self,
        lr: float = 0.001,
        momentum: float = 0,
        alpha: float = 0.99,
        eps: float = 1e-08,
        centered: bool = False,
        weight_decay: float = 0,
    ):
        super().__init__(lr)
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
