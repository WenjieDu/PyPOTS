"""
Data saving utilities with HDF5.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
from typing import Optional

import h5py

from pypots.utils.file import extract_parent_dir, create_dir_if_not_exist
from pypots.utils.logging import logger


def save_dict_into_h5(
    data_dict: dict,
    saving_path: str,
    file_name: Optional[str] = None,
) -> None:
    """Save the given data (in a dictionary) into the given h5 file.

    Parameters
    ----------
    data_dict : dict,
        The data to be saved, should be a Python dictionary.

    saving_path : str,
        If `file_name` is not given, the given path should be a path to a file with ".h5" suffix.
        If `file_name` is given, the given path should be a path to a directory.
        If parent directories don't exist, they will be created.

    file_name : str, optional (default=None)
        The name of the H5 file to be saved and should be with ".h5" suffix.
        It's optional. If not set, `saving_path` should be a path to a file with ".h5" suffix.

    """

    def save_set(handle, name, data):
        if isinstance(data, dict):
            single_set_handle = handle.create_group(name)
            for key, value in data.items():
                save_set(single_set_handle, key, value)
        else:
            handle.create_dataset(name, data=data)

    # check typing
    assert isinstance(
        data_dict, dict
    ), f"`data_dict` should be a Python dictionary, but got {type(data_dict)}."
    assert isinstance(
        saving_path, str
    ), f"`saving_path` should be a string, but got {type(saving_path)}."

    if file_name is None:  # if file_name is not given
        # check suffix
        if not saving_path.endswith(".h5") or saving_path.endswith(".hdf5"):
            logger.warning(
                f"‼️ `saving_path` should end with '.h5' or '.hdf5', but got {saving_path}. "
                f"PyPOTS will automatically append '.h5' to the given `saving_path`."
            )
    else:  # if file_name is given
        # check typing
        assert isinstance(
            file_name, str
        ), f"`file_name` should be a string, but got {type(file_name)}."
        # check suffix
        if not file_name.endswith(".h5") or file_name.endswith(".hdf5"):
            logger.warning(
                f"‼️ `file_name` should end with '.h5' or '.hdf5', but got {file_name}. "
                f"PyPOTS will automatically append '.h5' to the given `file_name`."
            )
        # organize the saving path
        saving_path = os.path.join(saving_path, file_name)

    # create the parent folders if not exist
    create_dir_if_not_exist(extract_parent_dir(saving_path))

    # create the h5 file handle and save the data
    with h5py.File(saving_path, "w") as hf:
        for k, v in data_dict.items():
            save_set(hf, k, v)

    logger.info(f"Successfully saved the given data into {saving_path}.")
