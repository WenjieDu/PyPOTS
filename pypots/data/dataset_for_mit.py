"""
Dataset class for self-attention models trained with MIT (masked imputation task) task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import torch
from pycorruptor import mcar

from pypots.data.base import BaseDataset


class DatasetForMIT(BaseDataset):
    """Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.

    For more information about MIT, please refer to :cite:`du2023SAITS`.

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

    file_type : str, default = "h5py"
        The type of the given file if train_set and val_set are path strings.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.
        Note that, `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.
    """

    def __init__(self, data, file_type="h5py", rate=0.2):
        super().__init__(data, file_type)
        self.rate = rate

    def _fetch_data_from_array(self, idx):
        """Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        sample : list,
            A list contains

            index : int tensor,
                The index of the sample.

            X_intact : tensor,
                Original time-series for calculating mask imputation loss.

            X : tensor,
                Time-series data with artificially missing values for model input.

            missing_mask : tensor,
                The mask records all missing values in X.

            indicating_mask : tensor.
                The mask indicates artificially missing values in X.
        """
        X = self.X[idx]
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            X_intact.to(torch.float32),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
            indicating_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx):
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx : int,
            The index of the sample to be return.

        Returns
        -------
        sample : list,
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx])
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            X_intact.to(torch.float32),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
            indicating_mask.to(torch.float32),
        ]

        if (
            "y" in self.file_handle.keys()
        ):  # if the dataset has labels, then fetch it from the file
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
