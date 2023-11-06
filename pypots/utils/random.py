"""
Transformer model for time-series imputation.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch
from .logging import logger

RANDOM_SEED = 2204


def set_random_seed(random_seed: int = RANDOM_SEED) -> None:
    """Manually set the random state to make PyPOTS output reproducible results.

    Parameters
    ----------
    random_seed :
        The seed to be set for generating random numbers in PyPOTS.

    """

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(random_seed)
    logger.info(f"Have set the random seed as {random_seed} for numpy and pytorch.")
