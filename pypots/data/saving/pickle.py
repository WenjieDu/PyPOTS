"""
Data saving utilities with pickle.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import pickle

import pandas as pd

from ...utils.file import extract_parent_dir, create_dir_if_not_exist
from ...utils.logging import logger


def pickle_dump(data: object, path: str) -> None:
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
        logger.info(f"Successfully saved to {path}")
    except Exception as e:
        logger.error(f"❌ Pickling failed. No cache data saved. Investigate the error below:\n{e}")

    return None


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
            if pd.__version__ >= "2.0.0":
                data = pd.read_pickle(f)
            else:
                data = pickle.load(f)
    except Exception as e:
        logger.error(f"❌ Loading data failed. Operation aborted. Investigate the error below:\n{e}")
        return None

    return data
