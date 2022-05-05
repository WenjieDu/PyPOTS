"""
Dataset class for model GRUD.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import torch

from pypots.data.base import BaseDataset
from pypots.data.dataset_for_brits import parse_delta
from pypots.imputation.locf import LOCF


class DatasetForGRUD(BaseDataset):
    """ Dataset class for model GRUD.

    Parameters
    ----------
    X : tensor, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super().__init__(X, y)

        self.locf = LOCF()
        self.missing_mask = (~torch.isnan(X)).to(torch.float32)
        self.X = torch.nan_to_num(X)
        self.deltas = parse_delta(self.missing_mask)
        self.X_filledLOCF = self.locf.locf_torch(X)
        self.empirical_mean = \
            torch.sum(self.missing_mask * self.X, dim=[0, 1]) / torch.sum(self.missing_mask, dim=[0, 1])

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
            sample.append(
                self.y[idx].to(torch.long)
            )

        return sample
