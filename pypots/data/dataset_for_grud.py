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
    """Dataset class for model GRUD.

    Parameters
    ----------
    X : tensor, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X=None, y=None, file_path=None, file_type="h5py"):
        super().__init__(X, y, file_path, file_type)

        self.locf = LOCF()

        if self.X is not None:
            self.missing_mask = (~torch.isnan(X)).to(torch.float32)
            self.X_filledLOCF = self.locf.locf_torch(X)
            self.X = torch.nan_to_num(X)
            self.deltas = parse_delta(self.missing_mask)
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

        if self.file_handler is None:
            self.file_handler = self._open_file_handle()

        X = torch.from_numpy(self.file_handler["X"][idx])
        missing_mask = (~torch.isnan(X)).to(torch.float32)
        X_filledLOCF = self.locf.locf_torch(X)
        X = torch.nan_to_num(X)
        deltas = parse_delta(missing_mask)
        empirical_mean = torch.sum(missing_mask * X, dim=[0, 1]) / torch.sum(
            missing_mask, dim=[0, 1]
        )

        sample = [
            torch.tensor(idx),
            X,
            X_filledLOCF,
            missing_mask,
            deltas,
            empirical_mean,
        ]

        if (
            "y" in self.file_handler.keys()
        ):  # if the dataset has labels, then fetch it from the file
            sample.append(self.file_handler["y"][idx].to(torch.long))

        return sample
