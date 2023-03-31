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
    missing_mask : tensor, shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, array,
        Delta matrix indicates time gaps of missing values.
        Its math definition please refer to :cite:`che2018GRUD`.
    """

    def cal_delta_for_single_sample(mask):
        d = []  # single sample's delta
        for step in range(n_steps):
            if step == 0:
                d.append(torch.zeros(1, n_features, device=device))
            else:
                d.append(
                    torch.ones(1, n_features, device=device) + (1 - mask[step]) * d[-1]
                )
        d = torch.concat(d, dim=0)
        return d

    # missing_mask is from X, and X's shape and type had been checked. So no need to double-check here.
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


def parse_delta_np(missing_mask):
    """Generate time-gap (delta) matrix from missing masks.

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

    def cal_delta_for_single_sample(mask):
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


class DatasetForBRITS(BaseDataset):
    """Dataset class for BRITS.

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

        if not isinstance(self.data, str):
            # calculate all delta here.
            forward_missing_mask = (~torch.isnan(self.X)).type(torch.float32)
            forward_X = torch.nan_to_num(self.X)
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

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx])
        missing_mask = (~torch.isnan(X)).to(torch.float32)
        X = torch.nan_to_num(X)

        forward = {
            "X": X,
            "missing_mask": missing_mask,
            "deltas": parse_delta(missing_mask),
        }

        backward = {
            "X": torch.flip(forward["X"], dims=[0]),
            "missing_mask": torch.flip(forward["missing_mask"], dims=[0]),
        }
        backward["deltas"] = parse_delta(backward["missing_mask"])

        sample = [
            torch.tensor(idx),
            # for forward
            forward["X"],
            forward["missing_mask"],
            forward["deltas"],
            # for backward
            backward["X"],
            backward["missing_mask"],
            backward["deltas"],
        ]

        if (
            "y" in self.file_handle.keys()
        ):  # if the dataset has labels, then fetch it from the file
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
