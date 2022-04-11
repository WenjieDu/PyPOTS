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


class Corruptor:
    """ Corrupt data by adding missing values to it with optional missing patterns (MCAR,MAR,MNAR).

    Parameters
    ----------
    data : array,
        Time series data.

    rate : float in (0,1),
        Artificially missing rate, rate of observed values which will be artificially masked as missing.

        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        Corruptor will do nothing, i.e. it won't play the role it has to be.

    Attributes
    ----------
    original_shape : tuple,
        The original shape of `data`.
    """

    def __init__(self, data, rate):
        self.data = data
        self.original_shape = data.shape
        self.rate = rate

    def originally_missing_rate(self):
        """ Calculate the originally missing rate of the raw data.

        Returns
        -------
        originally_missing_rate, float,
            The originally missing rate of the raw data.
        """
        originally_missing_rate = np.sum(np.isnan(self.data)) / np.product(self.data.shape)
        return originally_missing_rate

    @staticmethod
    def fill_nan_with_mask(data, mask):
        """ Fill missing values in `data` with nan according to mask.

        Parameters
        ----------
        data : array,
            Data vector having missing values filled with numbers (i.e. not nan).

        mask : array,
            Mask vector contains binary values indicating which values are missing in `data`.

        Returns
        -------
        array,
            Data vector having missing values placed with np.nan.
        """
        assert data.shape == mask.shape, f'Shapes of data and mask must match, ' \
                                         f'but data.shape={data.shape}, mask.shape={mask.shape}'
        mask = bool(mask)
        data[mask] = np.nan
        return data

    def mcar(self, nan=0):
        """ Create completely random missing values (MCAR case).

        Parameters
        ----------
        nan : int/float, optional,
            Value to be used to fill NaN values.

        Returns
        -------
        X_intact : array,
            Original data with missing values (nan) filled with given parameter `nan`, with observed values intact.
            X_intact is for loss calculation in the masked imputation task.
        X : array,
            Original X with artificial missing values. X is for model input.
            Both originally-missing and artificially-missing values are filled with given parameter `nan`.
        missing_mask : array,
            The mask indicates all missing values in X.
        indicating_mask : array,
            The mask indicates the artificially-missing values in X, namely missing parts different from X_intact.
        """
        X = self.data.flatten()
        # select random indices for artificial mask
        indices = np.where(~np.isnan(X))[0].tolist()  # get the indices of observed values
        indices = np.random.choice(indices, int(len(indices) * self.rate), replace=False)
        # create artificially-missing values by selected indices
        X[indices] = np.nan  # mask values selected by indices
        indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X))).astype(np.float32)
        missing_mask = (~np.isnan(X)).astype(np.float32)
        X = np.nan_to_num(X, nan=nan)
        X_intact = np.nan_to_num(self.data, nan=nan)
        # reshape into time-series data
        X = X.reshape(self.original_shape)
        missing_mask = missing_mask.reshape(self.original_shape)
        indicating_mask = indicating_mask.reshape(self.original_shape)
        return X_intact, X, missing_mask, indicating_mask

    def mar(self, nan=0):
        """ Create random missing values (MAR case).

        Parameters
        ----------
        nan : int/float, optional,
            Value to be used to fill NaN values.

        Returns
        -------

        """
        pass

    def mnar(self, nan=0):
        """ Create not-random missing values (MNAR case).

        Parameters
        ----------
        nan : int/float, optional,
            Value to be used to fill NaN values.

        Returns
        -------

        """
        pass


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
