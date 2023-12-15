"""
Data saving utilities with pickle.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import pickle
from typing import Optional

from pypots.utils.logging import logger


def pickle_dump(data: object, path: str) -> Optional[str]:
    """Pickle the given object.

    Parameters
    ----------
    data:
        The object to be pickled.

    path:
        Saving path.

    Returns
    -------
    `path` if succeed else None

    """
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError:
        logger.error("❌ Pickling failed. No cache data saved.")
        return None
    logger.info(f"Successfully saved to {path}")
    return path


def pickle_load(path: str) -> object:
    """Load pickled object from file.

    Parameters
    ----------
    path :
        Local path of the pickled object.

    Returns
    -------
    Object
        Pickled object.

    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except pickle.UnpicklingError as e:
        logger.error(
            "❌ Data file corrupted. Operation aborted. See info below:\n" f"{e}"
        )
    return data
