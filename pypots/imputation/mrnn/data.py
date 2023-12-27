"""
Dataset class for model MRNN.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import torch
from pygrinder import fill_and_get_mask_torch

from ...data.base import BaseDataset


def mrnn_parse_delta_torch(missing_mask: torch.Tensor) -> torch.Tensor:
    """Generate the time-gap matrix from the missing mask, this implementation is the same with the MRNN official
    implementation in tensorflow https://github.com/jsyoon0823/MRNN, but that is different from the description in the
    MRNN paper which is the same with the one from GRUD.

    In PyPOTS team's experiments, we find that this implementation is important to the training stability and
    the performance of MRNN, we think this is mainly because this version make the first step of deltas start from 1,
    rather than from 0 in the original description.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.
    """

    def cal_delta_for_single_sample(mask: torch.Tensor) -> torch.Tensor:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(n_steps):
            if step == 0:
                d.append(torch.ones(1, n_features, device=device))
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


class DatasetForMRNN(BaseDataset):
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

    return_labels : bool, default = True,
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type : str, default = "h5py"
        The type of the given file if train_set and val_set are path strings.
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_labels: bool,
        file_type: str = "h5py",
    ):
        super().__init__(data, return_X_ori, return_labels, file_type)

        if not isinstance(self.data, str):
            # calculate all delta here.
            if self.X_ori is None:
                forward_X, forward_missing_mask = fill_and_get_mask_torch(self.X)
            else:
                forward_missing_mask = self.missing_mask
                forward_X = self.X

            forward_delta = mrnn_parse_delta_torch(forward_missing_mask)
            backward_X = torch.flip(forward_X, dims=[1])
            backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])
            backward_delta = mrnn_parse_delta_torch(backward_missing_mask)

            self.processed_data = {
                "forward": {
                    "X": forward_X.to(torch.float32),
                    "missing_mask": forward_missing_mask.to(torch.float32),
                    "delta": forward_delta.to(torch.float32),
                },
                "backward": {
                    "X": backward_X.to(torch.float32),
                    "missing_mask": backward_missing_mask.to(torch.float32),
                    "delta": backward_delta.to(torch.float32),
                },
            }

    def _fetch_data_from_array(self, idx: int) -> Iterable:
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
            self.processed_data["forward"]["X"][idx],
            self.processed_data["forward"]["missing_mask"][idx],
            self.processed_data["forward"]["delta"][idx],
            # for backward
            self.processed_data["backward"]["X"][idx],
            self.processed_data["backward"]["missing_mask"][idx],
            self.processed_data["backward"]["delta"][idx],
        ]

        if self.X_ori is not None and self.return_X_ori:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])

        if self.y is not None and self.return_labels:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
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

        X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        X, missing_mask = fill_and_get_mask_torch(X)

        forward = {
            "X": X,
            "missing_mask": missing_mask,
            "deltas": mrnn_parse_delta_torch(missing_mask),
        }

        backward = {
            "X": torch.flip(forward["X"], dims=[0]),
            "missing_mask": torch.flip(forward["missing_mask"], dims=[0]),
        }
        backward["deltas"] = mrnn_parse_delta_torch(backward["missing_mask"])

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

        if "X_ori" in self.file_handle.keys() and self.return_X_ori:
            X_ori = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = X_ori_missing_mask - missing_mask
            sample.extend([X_ori, indicating_mask])

        # if the dataset has labels and is for training, then fetch it from the file
        if "y" in self.file_handle.keys() and self.return_labels:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
