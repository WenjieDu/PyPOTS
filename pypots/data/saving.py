"""
Data saving utilities.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os

import h5py

from ..utils.file import create_dir_if_not_exist
from ..utils.logging import logger


def save_dict_into_h5(
    data_dict: dict,
    saving_dir: str,
    saving_name: str = "datasets.h5",
) -> None:
    """Save the given data (in a dictionary) into the given h5 file.

    Parameters
    ----------
    data_dict : dict,
        The data to be saved, should be a Python dictionary.

    saving_dir : str,
        The h5 file to save the data.

    saving_name : str, optional (default="datasets.h5")
        The final name of the saved h5 file.

    """

    def save_set(handle, name, data):
        if isinstance(data, dict):
            single_set_handle = handle.create_group(name)
            for key, value in data.items():
                save_set(single_set_handle, key, value)
        else:
            handle.create_dataset(name, data=data)

    create_dir_if_not_exist(saving_dir)
    saving_path = os.path.join(saving_dir, saving_name)
    with h5py.File(saving_path, "w") as hf:
        for k, v in data_dict.items():
            save_set(hf, k, v)
    logger.info(f"Successfully saved the given data into {saving_path}.")
