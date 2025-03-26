"""
Dataset class for the imputation model SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import torch
from pygrinder import mcar, fill_and_get_mask_torch

from ...data.dataset.base import BaseDataset


class DatasetForSAITS(BaseDataset):
    """Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.

    For more information about MIT, please refer to :cite:`du2023SAITS`.

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

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.
        Note that, `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        rate: float = 0.2,
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=return_y,
            file_type=file_type,
        )
        self.rate = rate

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx :
            The index to fetch the specified sample.

        Returns
        -------
        sample :
            A list contains

            index :
                The index of the sample.

            X_ori :
                Original time-series for calculating mask imputation loss.

            X :
                Time-series data with artificially missing values for model input.

            missing_mask :
                The mask records all missing values in X.

            indicating_mask :
                The mask indicates artificially missing values in X.
        """

        if self.return_X_ori:
            X = self.X[idx]
            X_ori = self.X_ori[idx]
            missing_mask = self.missing_mask[idx]
            indicating_mask = self.indicating_mask[idx]
        else:
            X_ori = self.X[idx]
            X = mcar(X_ori, p=self.rate)
            X, missing_mask = fill_and_get_mask_torch(X)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)

        sample = [
            torch.tensor(idx),
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ]

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        if self.return_X_ori:
            X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            X, missing_mask = fill_and_get_mask_torch(X)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)
        else:
            X_ori = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X = mcar(X_ori, p=self.rate)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            X, missing_mask = fill_and_get_mask_torch(X)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)

        sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
