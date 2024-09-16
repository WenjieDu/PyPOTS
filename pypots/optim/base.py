"""
The base wrapper for PyTorch optimizers (https://pytorch.org/docs/stable/optim.html#algorithms),
also is the base class for all optimizers in pypots.optim.

The optimizers in pypots.optim are all wrappers for PyTorch optimizers.
pypots.optim.optimizers inherent all functionalities from torch.optim.optimizers (so you can see many docstrings
are copied from torch), but are more powerful. So far, they are designed to:

1). separate the hyperparameters of models and optimizers in PyPOTS, so that users don't have to put all hyperparameters
in one place, which could result in a mess and be not readable;

2). provide additional functionalities, such as learning rate scheduling, etc.;

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional

from .lr_scheduler.base import LRScheduler


class Optimizer(ABC):
    """The base wrapper for PyTorch optimizers, also is the base class for all optimizers in PyPOTS.

    Parameters
    ----------
    lr : float
        The learning rate of the optimizer.

    lr_scheduler : pypots.optim.lr_scheduler.base.LRScheduler
        The learning rate scheduler of the optimizer.

    Attributes
    ----------
    torch_optimizer :
        The torch optimizer wrapped by this class.

    """

    def __init__(self, lr, lr_scheduler: Optional[LRScheduler] = None):
        self.lr = lr
        self.torch_optimizer = None
        self.lr_scheduler = lr_scheduler

    @abstractmethod
    def init_optimizer(self, params: Iterable) -> None:
        """Initialize the torch optimizer wrapped by this class.

        Parameters
        ----------
        params :
            An iterable of ``torch.Tensor`` or ``dict``. Specifies what Tensors should be optimized.
        """
        raise NotImplementedError

    def add_param_group(self, param_group: dict) -> None:
        """Add a param group to the optimizer param_groups.

        Parameters
        ----------
        param_group :
            Specifies the parameters to be optimized and group-specific optimization options.
        """
        self.torch_optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict) -> None:
        """Loads the optimizer state.

        Parameters
        ----------
        state_dict :
            Optimizer state. It should be an object returned from ``state_dict()``.
        """

        self.torch_optimizer.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        """Returns the state of the optimizer as a dict.

        Returns
        -------
        state_dict :
            The state dict of the optimizer, which contains two entries:
            1). state - a dict holding current optimization state. Its content differs between optimizer classes.
            2). param_groups - a list containing all parameter groups where each parameter group is a dict

        """
        state_dict = self.torch_optimizer.state_dict()
        return state_dict

    def step(self, closure: Optional[Callable] = None) -> None:
        """Performs a single optimization step (parameter update).

        Parameters
        ----------
        closure :
            A closure that reevaluates the model and returns the loss. Optional for most optimizers.
            Refer to the :class:`torch.optim.Optimizer.step()` docstring for more details.

        """
        self.torch_optimizer.step(closure)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Sets the gradients of all optimized ``torch.Tensor`` to zero.

        Parameters
        ----------
        set_to_none :
            Instead of setting to zero, set the grads to None.
            Refer to the torch.optim.Optimizer.zero_grad() docstring for more details.

        """
        self.torch_optimizer.zero_grad(set_to_none)
