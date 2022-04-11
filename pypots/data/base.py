"""
Utilities for data manipulation
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from pypots.data.corrupt import mcar


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


class Dataset4BRITS(BaseDataset):
    """ Dataset class for BRITS.

    Parameters
    ----------
    X : array-like, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.
    y : array-like, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super(Dataset4BRITS, self).__init__(X, y)

    @staticmethod
    def parse_delta(missing_mask):
        """ Generate time-gap (delta) matrix from missing masks.

        Parameters
        ----------
        missing_mask : array, shape of [seq_len, n_features]
            Binary masks indicate missing values.

        Returns
        -------
        delta, array,
            Delta matrix indicates time gaps of missing values.
            Its math definition please refer to :cite:`che2018MissingData`.
        """

        assert len(missing_mask.shape) == 2, f'missing_mask should has two dimensions, ' \
                                             f'shape like [seq_len, n_features], ' \
                                             f'while the input is {missing_mask.shape}'
        seq_len, n_features = missing_mask.shape
        delta = []
        for step in range(seq_len):
            if step == 0:
                delta.append(np.zeros(n_features))
            else:
                delta.append(np.ones(n_features) + (1 - missing_mask[step]) * delta[-1])
        return np.asarray(delta)

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
            delta : tensor,
                The delta matrix contains time gaps of missing values.
            label (optional) : tensor,
                The target label of the time-series sample.
        """
        X = self.X[idx]
        missing_mask = (~np.isnan(X)).astype(np.float32)
        X = np.nan_to_num(X)

        forward = {
            'X': X,
            'missing_mask': missing_mask,
            'deltas': self.parse_delta(missing_mask)
        }
        backward = {
            'X': np.flip(forward['X'], axis=0).copy(),
            'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()
        }
        backward['deltas'] = self.parse_delta(backward['missing_mask'])

        sample = [
            torch.tensor(idx),
            # for forward
            torch.from_numpy(forward['X'].astype('float32')),
            torch.from_numpy(forward['missing_mask'].astype('float32')),
            torch.from_numpy(forward['deltas'].astype('float32')),
            # for backward
            torch.from_numpy(backward['X'].astype('float32')),
            torch.from_numpy(backward['missing_mask'].astype('float32')),
            torch.from_numpy(backward['deltas'].astype('float32')),
        ]

        if self.y is not None:
            sample.append(torch.from_numpy(self.y[idx]))

        return sample


class Dataset4MIT(BaseDataset):
    """ Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.

    For more information about MIT, please refer to :cite:`du2022SAITS`.

    Parameters
    ----------
    X : array-like, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.

    y : array-like, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.

        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.

    """

    def __init__(self, X, y=None, rate=0.2):
        super(Dataset4MIT, self).__init__(X, y)
        self.rate = rate

    def __getitem__(self, idx):
        X = self.X[idx]
        X_intact, X, missing_mask, indicating_mask = mcar(X, rate=self.rate)

        sample = [
            torch.tensor(idx),
            torch.from_numpy(X_intact.astype('float32')),
            torch.from_numpy(X.astype('float32')),
            torch.from_numpy(missing_mask.astype('float32')),
            torch.from_numpy(indicating_mask.astype('float32')),
        ]

        if self.y is not None:
            sample.append(torch.from_numpy(self.y[idx]))

        return sample
