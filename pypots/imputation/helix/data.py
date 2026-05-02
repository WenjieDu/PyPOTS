"""
Dataset class for the imputation model HELIX.
"""

# Created by MiBah Cat <milaogou@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import torch
from pygrinder import mcar, fill_and_get_mask_torch

from ...data.dataset.base import BaseDataset


class DatasetForHELIX(BaseDataset):
    """Dataset for HELIX model that needs MIT (masked imputation task) in training.

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

    return_X_ori :
        Whether to return original data (X_ori) in __getitem__().

    return_y :
        Whether to return labels in function __getitem__() if they exist in the given data.

    file_type :
        The type of the given file if train_set and val_set are path strings.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.
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
            A list containing index, X, missing_mask, X_ori, indicating_mask, and optionally y.
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
        """Fetch data with the lazy-loading strategy.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            The collated data sample.
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

        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample