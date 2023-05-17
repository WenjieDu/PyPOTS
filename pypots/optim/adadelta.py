"""
The optimizer wrapper for PyTorch Adadelta.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Iterable

from torch.optim import Adadelta as torch_Adadelta

from pypots.optim.base import Optimizer


class Adadelta(Optimizer):
    """The optimizer wrapper for PyTorch Adadelta.
    https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta

    Parameters
    ----------
    lr : float, default=0.01
        The learning rate of the optimizer.

    rho : float, default=0.9
        Coefficient used for computing a running average of squared gradients.

    eps : float, default=1e-08
        Term added to the denominator to improve numerical stability.

    weight_decay : float, default=0.01
        Weight decay (L2 penalty).

    """

    def __init__(
        self,
        lr: float = 0.01,
        rho: float = 0.9,
        eps: float = 1e-08,
        weight_decay: float = 0.01,
    ):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params : Iterable,
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.
        """
        self.torch_optimizer = torch_Adadelta(
            params=params,
            lr=self.lr,
            rho=self.rho,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
