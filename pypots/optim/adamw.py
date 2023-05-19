"""
The optimizer wrapper for PyTorch AdamW.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Iterable, Tuple

from torch.optim import AdamW as torch_AdamW

from .base import Optimizer


class AdamW(Optimizer):
    """The optimizer wrapper for PyTorch AdamW.
    https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW

    Parameters
    ----------
    lr :
        The learning rate of the optimizer.

    betas :
        Coefficients used for computing running averages of gradient and its square.

    eps :
        Term added to the denominator to improve numerical stability.

    weight_decay :
        Weight decay (L2 penalty).

    amsgrad :
        Whether to use the AMSGrad variant of this algorithm from the paper :cite:`reddi2018OnTheConvergence`.
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        super().__init__(lr)
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
