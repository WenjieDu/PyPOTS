"""
Utilities for data manipulation
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):
    """Base dataset class in PyPOTS.

    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series feature vector.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super().__init__()
        # types and shapes had been checked after X and y input into the model
        # So they are safe to use here. No need to check again.
        self.X = X
        self.y = y
        self.n_steps = self.X.shape[1]
        self.n_features = self.X.shape[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.
        """
        X = self.X[idx]
        missing_mask = ~torch.isnan(X)
        X = torch.nan_to_num(X)

        sample = [
            torch.tensor(idx),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(self.y[idx].to(torch.long))

        return sample
