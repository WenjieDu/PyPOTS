"""
Dataset class for model GRU-D.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import numpy as np
import torch

from pypots.data.base import BaseDataset
from pypots.imputation.locf import LOCF


def torch_parse_delta(missing_mask):
    """Generate time-gap (delta) matrix from missing masks. Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : torch.tensor, shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, torch.tensor,
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask):
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(n_steps):
            if step == 0:
                d.append(torch.zeros(1, n_features, device=device))
            else:
                d.append(
                    torch.ones(1, n_features, device=device) + (1 - mask[step]) * d[-1]
                )
        d = torch.concat(d, dim=0)
        return d

    device = missing_mask.device
    if len(missing_mask.shape) == 2:
        n_steps, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, n_steps, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta.unsqueeze(0))
        delta = torch.concat(delta_collector, dim=0)

    return delta


def numpy_parse_delta(missing_mask):
    """Generate time-gap (delta) matrix from missing masks. Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : np.ndarray, shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, np.ndarray,
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask):
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(seq_len):
            if step == 0:
                d.append(np.zeros(n_features))
            else:
                d.append(np.ones(n_features) + (1 - mask[step]) * d[-1])
        d = np.asarray(d)
        return d

    if len(missing_mask.shape) == 2:
        seq_len, n_features = missing_mask.shape
        delta = cal_delta_for_single_sample(missing_mask)
    else:
        n_samples, seq_len, n_features = missing_mask.shape
        delta_collector = []
        for m_mask in missing_mask:
            delta = cal_delta_for_single_sample(m_mask)
            delta_collector.append(delta)
        delta = np.asarray(delta_collector)
    return delta


class DatasetForGRUD(BaseDataset):
    """Dataset class for model GRUD.

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
    """

    def __init__(self, data, file_type="h5py"):
        super().__init__(data, file_type)

        self.locf = LOCF()

        if not isinstance(self.data, str):  # data from array
            self.missing_mask = (~torch.isnan(self.X)).to(torch.float32)
            self.X_filledLOCF = self.locf.locf_torch(self.X)
            self.X = torch.nan_to_num(self.X)
            self.deltas = torch_parse_delta(self.missing_mask)
            self.empirical_mean = torch.sum(
                self.missing_mask * self.X, dim=[0, 1]
            ) / torch.sum(self.missing_mask, dim=[0, 1])

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

            X : tensor,
                The feature vector for model input.

            X_filledLOCF: tensor,
                The feature vector filled with last observations.

            missing_mask : tensor,
                The mask indicates all missing values in X.

            delta : tensor,
                The delta matrix contains time gaps of missing values.

            empirical_mean : tensor,
                Mean values of features.
        """
        sample = [
            torch.tensor(idx),
            self.X[idx].to(torch.float32),
            self.X_filledLOCF[idx].to(torch.float32),
            self.missing_mask[idx].to(torch.float32),
            self.deltas[idx].to(torch.float32),
            self.empirical_mean.to(torch.float32),
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
        missing_mask = (~torch.isnan(X)).to(torch.float32)
        X_filledLOCF = self.locf.locf_torch(X.unsqueeze(dim=0)).squeeze()
        X = torch.nan_to_num(X)
        deltas = torch_parse_delta(missing_mask)
        empirical_mean = torch.sum(missing_mask * X, dim=[0]) / torch.sum(
            missing_mask, dim=[0]
        )

        sample = [
            torch.tensor(idx),
            X,
            X_filledLOCF,
            missing_mask,
            deltas,
            empirical_mean,
        ]

        # if the dataset has labels, then fetch it from the file
        if "y" in self.file_handle.keys():
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
