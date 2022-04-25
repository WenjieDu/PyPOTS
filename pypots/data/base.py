"""
Utilities for data manipulation
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np

from torch.utils.data import Dataset


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
        super().__init__()
        assert isinstance(X, np.ndarray), f'X should be numpy array, but got {type(X)}'
        assert len(X.shape) == 3, "X should have 3 dimensions, [n_samples, seq_len, n_features]."
        if y is not None:
            assert isinstance(y, np.ndarray), f'y should be numpy array, but got {type(y)}'

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
        """
        pass
