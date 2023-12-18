"""
Dataset class for model USGAN.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

from ..brits.data import DatasetForBRITS


class DatasetForUSGAN(DatasetForBRITS):
    """Dataset class for USGAN, the same with the one for BRITS.

    Parameters
    ----------
    data : dict or str,
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    return_labels : bool, default = True,
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type : str, default = "h5py"
        The type of the given file if train_set and val_set are path strings.
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_labels: bool,
        file_type: str = "h5py",
    ):
        super().__init__(data, return_X_ori, return_labels, file_type)
