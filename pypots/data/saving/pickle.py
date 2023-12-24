"""
Data saving utilities with pickle.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import pickle
from typing import Optional

from ...utils.file import extract_parent_dir, create_dir_if_not_exist
from ...utils.logging import logger


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
        # help create the parent dir if not exist
        create_dir_if_not_exist(extract_parent_dir(path))
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(
            f"❌ Pickling failed. No cache data saved. Please investigate the error below.\n{e}"
        )
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
    except Exception as e:
        logger.error(f"❌ Loading data failed. Operation aborted. See info below:\n{e}")
    return data
