"""
Utilities for checking things.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os

from .logging import logger


def extract_parent_dir(path: str) -> str:
    """Extract the given path's parent directory.

    Parameters
    ----------
    path :
        The path for extracting.

    Returns
    -------
    parent_dir :
        The path to the parent dir of the given path.

    """
    parent_dir = os.path.abspath(os.path.join(path, ".."))
    return parent_dir


def create_dir_if_not_exist(path: str, is_dir: bool = True) -> None:
    """Create the given directory if it doesn't exist.

    Parameters
    ----------
    path :
        The path for check.

    is_dir :
        Whether the given path is to a directory. If `is_dir` is False, the given path is to a file or an object,
        then this file's parent directory will be checked.

    """
    path = extract_parent_dir(path) if not is_dir else path
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Successfully created the given path {path}")


def get_class_full_path(cls) -> str:
    """Get the full path of the given class.

    Parameters
    ----------
    cls:
        The class to get the full path.

    Returns
    -------
    path :
        The full path of the given class.

    """
    module = cls.__module__
    path = cls.__qualname__
    if module is not None and module != "__builtin__":
        path = module + "." + path
    return path
