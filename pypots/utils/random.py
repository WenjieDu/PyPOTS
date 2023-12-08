"""
PyPOTS util module about random seed setting.
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
    globals()["RANDOM_SEED"] = random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    logger.info(f"Have set the random seed as {random_seed} for numpy and pytorch.")


def get_random_seed() -> int:
    """Get the random seed used in PyPOTS.

    Returns
    -------
    random_seed :
        The random seed used in PyPOTS.

    """
    return RANDOM_SEED
