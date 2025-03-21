"""
The base class for learning rate schedulers. This class is adapted from PyTorch,
please refer to torch.optim.lr_scheduler for more details.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import weakref
from abc import ABC, abstractmethod
from functools import wraps

from torch.optim import Optimizer

from ...utils.logging import logger


class LRScheduler(ABC):
    """Base class for PyPOTS learning rate schedulers.

    Parameters
    ----------
    last_epoch: int
        The index of last epoch. Default: -1.

    verbose: If ``True``, prints a message to stdout for
        each update. Default: ``False``.

    """

    def __init__(self, last_epoch=-1, verbose=False):
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.optimizer = None
        self.base_lrs = None
        self._last_lr = None
        self._step_count = 0

    def init_scheduler(self, optimizer):
        """Initialize the scheduler. This method should be called in
        :class:`pypots.optim.base.Optimizer.init_optimizer()` to initialize the scheduler together with the optimizer.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            The optimizer to be scheduled.

        """

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)

        # Initialize epoch and base learning rates
        if self.last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0

    @abstractmethod
    def get_lr(self):
        """Compute learning rate."""
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    @staticmethod
    def print_lr(is_verbose, group, lr):
        """Display the current learning rate."""
        if is_verbose:
            logger.info(f"Adjusting learning rate of group {group} to {lr:.4e}.")

    def step(self):
        """Step could be called after every batch update.
        This should be called in :class:`pypots.optim.base.Optimizer.step()` after
        :class:`pypots.optim.base.Optimizer.torch_optimizer.step()`.
        """
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                logger.warning(
                    "⚠️ Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                logger.warning(
                    "⚠️ Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                )
        self._step_count += 1

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            self.last_epoch += 1
            values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
