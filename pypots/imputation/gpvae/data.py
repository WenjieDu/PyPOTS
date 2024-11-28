"""
Dataset class for the imputation model GP-VAE.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import torch
from pygrinder import fill_and_get_mask_torch

from ...data.dataset import BaseDataset


class DatasetForGPVAE(BaseDataset):
    """Dataset class for GP-VAE.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
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
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=return_y,
            file_type=file_type,
        )

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            X : tensor,
                The feature vector for model input.

            missing_mask : tensor,
                The mask indicates all missing values in X.

            delta : tensor,
                The delta matrix contains time gaps of missing values.

            label (optional) : tensor,
                The target label of the time-series sample.
        """
        X = self.X[idx]

        if self.return_X_ori:
            X = self.X[idx]
            missing_mask = self.missing_mask[idx]
            X_ori = self.X_ori[idx]
            indicating_mask = self.indicating_mask[idx]
            sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]
        else:
            X, missing_mask = fill_and_get_mask_torch(X)
            sample = [torch.tensor(idx), X, missing_mask]

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
            sample = [torch.tensor(idx), X, missing_mask, X_ori, indicating_mask]
        else:
            X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            X, missing_mask = fill_and_get_mask_torch(X)
            sample = [torch.tensor(idx), X, missing_mask]

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
