"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union

import h5py


def key_in_data_set(key: str, dataset: Union[str, dict]) -> bool:
    """Check if the key is in the given dataset.
    The dataset could be a path to an HDF5 file or a Python dictionary.

    Parameters
    ----------
    key :
        The key to check.

    dataset :
        The dataset to be checked.

    Returns
    -------
    bool
        Whether the key is in the dataset.
    """

    if isinstance(dataset, str):
        with h5py.File(dataset, "r") as f:
            return key in f.keys()
    elif isinstance(dataset, dict):
        return key in dataset.keys()
    else:
        raise TypeError(f"dataset must be a str or a Python dictionary, but got {type(dataset)}")
