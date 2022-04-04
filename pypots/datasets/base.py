"""
Utilities for data manipulation
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset


def generate_random_walk(n_samples=1000, n_steps=24, n_features=10, mu=0., std=1., random_state=None):
    """
    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.
    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.
    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.
    mu : float, default 0.0,
        Mean of the normal distribution, which random walk steps are sampled from.
    std : float, default 1.,
        Standard deviation of the normal distribution, which random walk steps are sampled from.
    random_state : int or numpy.RandomState, default=None,
        Random seed for data generation.

    Returns
    -------
    array, shape of [n_samples, n_steps, n_features]
        Generated random walk time series.
    """
    seed = check_random_state(random_state)
    ts_samples = np.empty((n_samples, n_steps, n_features))
    noise = seed.randn(n_samples, n_steps, n_features) * std + mu
    ts_samples[:, 0, :] = noise[:, 0, :]
    for t in range(1, n_steps):
        ts_samples[:, t, :] = ts_samples[:, t - 1, :] + noise[:, t, :]
    return ts_samples


def random_mask(X, artificial_missing_rate, nan=0):
    """ Randomly mask out some observed values to create artificially-missing values.
    This step is for training model on the masked imputation task (MIT).
    Please refer to :cite:`du2022SAITS` for more details about MIT.

    Parameters
    ----------
    X : array,
        Input feature vector. If X has any missing values, they should be numpy.nan.
    artificial_missing_rate : float in (0,1),
        Rate of observed values which will be artificially masked as missing.
    nan : int/float, optional,
        Value to be used to fill NaN values.

    Returns
    -------
    X_intact : array,
        Original X with missing values (nan) filled with given parameter `nan`, with observed values intact.
        X_intact is for loss calculation in the masked imputation task.
    X : array,
        Original X with artificial missing values. X is for model input.
        Both originally-missing and artificially-missing values are filled with given parameter `nan`.
    missing_mask : array,
        The mask indicates all missing values in X.
    indicating_mask : array,
        The mask indicates the artificially-missing values in X, namely missing parts different from X_intact.
    """
    original_shape = X.shape
    X = X.reshape(-1)
    # select random indices for artificial mask
    indices = np.where(~np.isnan(X))[0].tolist()  # get the indices of observed values
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate), replace=False)
    # create artificially-missing values by selected indices
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    X[indices] = np.nan  # mask values selected by indices
    indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X = np.nan_to_num(X, nan=nan)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    # reshape into time-series data
    X_intact = X_intact.reshape(original_shape)
    X = X.reshape(original_shape)
    missing_mask = missing_mask.reshape(original_shape)
    indicating_mask = indicating_mask.reshape(original_shape)
    return X_intact, X, missing_mask, indicating_mask


class BaseDataset(Dataset):
    """ Base dataset class in PyPOTS.

    Parameters
    ----------
    X : array-like, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.
    y : array-like, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super(BaseDataset, self).__init__()
        assert len(X.shape) == 3, "X should have 3 dimensions, [n_samples, seq_len, n_features]."
        self.X = X
        self.y = y
        self.seq_len = self.X.shape[1]
        self.n_features = self.X.shape[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """ Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        dict,
            A dict contains
            index : int tensor,
                The index of the sample.
            X : tensor,
                The feature vector for model input.
            missing_mask : tensor,
                The mask indicates all missing values in X.
            label (optional) : tensor,
                The target label of the time-series sample.
        """
        X_intact = self.X[idx]

        missing_mask = (~np.isnan(X_intact)).astype(np.float32)
        X_intact = np.nan_to_num(X_intact)
        sample = {
            'index': torch.tensor(idx),
            'X': torch.from_numpy(X_intact),  # no artificially missing values, so X_intact is for model input
            'missing_mask': torch.from_numpy(missing_mask),
        }

        if self.y:
            sample['label'] = torch.from_numpy(self.y[idx])

        return sample
