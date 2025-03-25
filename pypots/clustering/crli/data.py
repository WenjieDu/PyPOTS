"""
Dataset class for the clustering model CRLI.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union, Iterable

from ...data.dataset.base import BaseDataset


class DatasetForCRLI(BaseDataset):
    """Dataset class for model CRLI.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    return_y :
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

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

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        return super()._fetch_data_from_array(idx)

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        return super()._fetch_data_from_file(idx)
