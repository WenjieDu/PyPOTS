"""
Utilities for checking things.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os

from pypots.utils.logging import logger


def extract_parent_dir(path):
    """Extract the given path's parent directory.

    Parameters
    ----------
    path : str,
        The path for extracting.

    Returns
    -------
    parent_dir : str
        The path to the parent dir of the given path.

    """
    parent_dir = os.path.abspath(os.path.join(path, ".."))
    return parent_dir


def create_dir_if_not_exist(path, is_dir=True):
    """Create the given directory if it doesn't exist.

    Parameters
    ----------
    path : str,
        The path for check.

    is_dir : bool,
        Whether the given path is to a directory. If `is_dir` is False, the given path is to a file or an object,
        then this file's parent directory will be checked.

    """
    path = extract_parent_dir(path) if not is_dir else path
    if os.path.exists(path):
        logger.info(f'The given directory "{path}" exists.')
    else:
        os.makedirs(path, exist_ok=True)
        logger.info(f'Successfully created "{path}".')
