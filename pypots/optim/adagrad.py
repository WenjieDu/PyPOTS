"""
The optimizer wrapper for PyTorch Adagrad.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Iterable

from torch.optim import Adagrad as torch_Adagrad

from .base import Optimizer


class Adagrad(Optimizer):
    """The optimizer wrapper for PyTorch Adagrad.
    https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad

    Parameters
    ----------
    lr :
        The learning rate of the optimizer.

    lr_decay :
        Learning rate decay.

    weight_decay :
        Weight decay (L2 penalty).

    eps :
        Term added to the denominator to improve numerical stability.

    initial_accumulator_value :
        A floating point value. Starting value for the accumulators, must be positive.

    """

    def __init__(
        self,
        lr: float = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0.01,
        initial_accumulator_value: float = 0.01,  # it is set as 0 in the torch implementation, but delta shouldn't be 0
        eps: float = 1e-08,
    ):
        super().__init__(lr)
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
