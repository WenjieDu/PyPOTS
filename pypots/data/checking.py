"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union

import h5py


def check_x_intact_in_val_set(val_set: Union[str, dict]) -> bool:
    if isinstance(val_set, str):
        with h5py.File(val_set, "r") as f:
            return "X_intact" in f.keys()
    elif isinstance(val_set, dict):
        return "X_intact" in val_set.keys()
    else:
        raise TypeError("val_set must be a str or a Python dictionary.")
