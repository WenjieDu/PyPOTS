"""
Dataset class for model BRITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import numpy as np
import torch

from pypots.data.base import BaseDataset


def parse_delta(missing_mask):
    """Generate time-gap (delta) matrix from missing masks.

    Parameters
    ----------
    missing_mask : tensor, shape of [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, array,
        Delta matrix indicates time gaps of missing values.
        Its math definition please refer to :cite:`che2018MissingData`.
    """
    # missing_mask is from X, and X's shape and type had been checked. So no need to double-check here.
    n_samples, n_steps, n_features = missing_mask.shape
    device = missing_mask.device
    delta_collector = []
    for m_mask in missing_mask:
        delta = []
        for step in range(n_steps):
            if step == 0:
                delta.append(torch.zeros(1, n_features, device=device))
            else:
                delta.append(
                    torch.ones(1, n_features, device=device)
                    + (1 - m_mask[step]) * delta[-1]
                )
        delta = torch.concat(delta, dim=0)
        delta_collector.append(delta.unsqueeze(0))
    delta = torch.concat(delta_collector, dim=0)
    return delta


class DatasetForBRITS(BaseDataset):
    """Dataset class for BRITS.

    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series data.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X=None, y=None, file_path=None, file_type="h5py"):
        super().__init__(X, y, file_path, file_type)

        if self.X is not None:
            # calculate all delta here.
            # Training will take too much time if we put delta calculation in __getitem__().
            forward_missing_mask = (~torch.isnan(X)).type(torch.float32)
            forward_X = torch.nan_to_num(X)
            forward_delta = parse_delta(forward_missing_mask)
            backward_X = torch.flip(forward_X, dims=[1])
            backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
            backward_delta = parse_delta(backward_missing_mask)

            self.processed_data = {
                "forward": {
                    "X": forward_X,
                    "missing_mask": forward_missing_mask,
                    "delta": forward_delta,
                },
                "backward": {
                    "X": backward_X,
                    "missing_mask": backward_missing_mask,
                    "delta": backward_delta,
                },
            }

    def _fetch_data_from_array(self, idx):
        """Fetch data from self.X if it is given.

        Parameters
        ----------
        idx : int,
            The index of the sample to be return.

        Returns
        -------
        sample : list,
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
        sample = [
            torch.tensor(idx),
            # for forward
            self.processed_data["forward"]["X"][idx].to(torch.float32),
            self.processed_data["forward"]["missing_mask"][idx].to(torch.float32),
            self.processed_data["forward"]["delta"][idx].to(torch.float32),
            # for backward
            self.processed_data["backward"]["X"][idx].to(torch.float32),
            self.processed_data["backward"]["missing_mask"][idx].to(torch.float32),
            self.processed_data["backward"]["delta"][idx].to(torch.float32),
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

        X = self.file_handler["X"][idx]
        missing_mask = (~np.isnan(X)).astype("float32")
        X = np.nan_to_num(X)

        forward = {
            "X": X,
            "missing_mask": missing_mask,
            "deltas": parse_delta(missing_mask),
        }

        backward = {
            "X": np.flip(forward["X"], axis=0).copy(),
            "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
        }
        backward["deltas"] = parse_delta(backward["missing_mask"])

        sample = [
            torch.tensor(idx),
            # for forward
            torch.from_numpy(forward["X"].astype("float32")),
            torch.from_numpy(forward["missing_mask"].astype("float32")),
            torch.from_numpy(forward["deltas"].astype("float32")),
            # for backward
            torch.from_numpy(backward["X"].astype("float32")),
            torch.from_numpy(backward["missing_mask"].astype("float32")),
            torch.from_numpy(backward["deltas"].astype("float32")),
        ]

        if (
            "y" in self.file_handler.keys()
        ):  # if the dataset has labels, then fetch it from the file
            sample.append(self.file_handler["y"][idx].to(torch.long))

        return sample
