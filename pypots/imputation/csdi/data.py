"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import torch
from pygrinder import mcar

from ...data.base import BaseDataset


class DatasetForCSDI(BaseDataset):
    """Dataset for CSDI model."""

    def __init__(
        self,
        data: Union[dict, str],
        return_labels: bool = True,
        file_type: str = "h5py",
        rate: float = 0.1,
    ):
        super().__init__(data, return_labels, file_type)
        self.time_points = (
            None if "time_points" not in data.keys() else data["time_points"]
        )
        # _, self.time_points = self._check_input(self.X, time_points)
        self.for_pattern_mask = (
            None if "for_pattern_mask" not in data.keys() else data["for_pattern_mask"]
        )
        # _, self.for_pattern_mask = self._check_input(self.X, for_pattern_mask)
        self.cut_length = (
            None if "cut_length" not in data.keys() else data["cut_length"]
        )
        # _, self.cut_length = self._check_input(self.X, cut_length)
        self.rate = rate

    def _fetch_data_from_array(self, idx: int) -> Iterable:
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

            X_intact : tensor,
                Original time-series for calculating mask imputation loss.

            X : tensor,
                Time-series data with artificially missing values for model input.

            missing_mask : tensor,
                The mask records all missing values in X.

            indicating_mask : tensor.
                The mask indicates artificially missing values in X.
        """
        X = self.X[idx].to(torch.float32)
        X_intact, X, missing_mask, indicating_mask = mcar(X, p=self.rate)

        observed_data = X_intact
        observed_mask = missing_mask + indicating_mask
        observed_tp = (
            torch.arange(0, self.n_steps, dtype=torch.float32)
            if self.time_points is None
            else self.time_points[idx].to(torch.float32)
        )
        gt_mask = indicating_mask
        for_pattern_mask = (
            gt_mask if self.for_pattern_mask is None else self.for_pattern_mask[idx]
        )
        cut_length = (
            torch.zeros(len(observed_data)).long()
            if self.cut_length is None
            else self.cut_length[idx]
        )

        sample = [
            torch.tensor(idx),
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        ]

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
        X_intact, X, missing_mask, indicating_mask = mcar(X, p=self.rate)

        observed_data = X_intact
        observed_mask = missing_mask + indicating_mask
        observed_tp = self.time_points[idx].to(torch.float32)
        gt_mask = indicating_mask
        for_pattern_mask = (
            gt_mask if self.for_pattern_mask is None else self.for_pattern_mask[idx]
        )
        cut_length = (
            torch.zeros(len(observed_data)).long()
            if self.cut_length is None
            else self.cut_length[idx]
        )

        sample = [
            torch.tensor(idx),
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        ]

        # if the dataset has labels and is for training, then fetch it from the file
        if "y" in self.file_handle.keys() and self.return_labels:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
