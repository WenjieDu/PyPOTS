"""
Dataset class for the vectorizer model TEFN.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ...data.dataset import BaseDataset


class DatasetForTEFN(BaseDataset):
    """Dataset class for TEFN.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is vectorization labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    file_type :
        The type of the given file if train_set and val_set are path strings.
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_y: bool = True,
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=False,
            return_X_pred=False,
            return_y=return_y,
            file_type=file_type,
        )
