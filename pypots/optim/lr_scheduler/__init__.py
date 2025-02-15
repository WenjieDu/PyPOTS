"""
Learning rate schedulers available in PyPOTS. Their functionalities are the same with those in PyTorch,
the only difference that is also why we implement them is that you don't have to pass according optimizers
into them immediately while initializing them. Instead, you can pass them into :class:`pypots.optim.base.Optimizer`
after initialization and call their `init_scheduler()` method in :class:`pypots.optim.base.Optimizer.init_optimizer()`
to initialize schedulers together with optimizers.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .constant_lrs import ConstantLR
from .exponential_lrs import ExponentialLR
from .lambda_lrs import LambdaLR
from .linear_lrs import LinearLR
from .multiplicative_lrs import MultiplicativeLR
from .multistep_lrs import MultiStepLR
from .step_lrs import StepLR

__all__ = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "ExponentialLR",
    "LinearLR",
]
