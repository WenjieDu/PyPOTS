"""
Dataset class for the imputation model CSDI.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import numpy as np
import torch
from pygrinder import fill_and_get_mask_torch

from ...data.dataset.base import BaseDataset


class DatasetForCSDI(BaseDataset):
    """Dataset for CSDI model.

    Notes
    -----
    In CSDI official code, `observed_mask` indicates all observed values in raw data.
    `gt_mask` indicates all observed values in the input data.
    `observed_mask` - `gt_mask` = `indicating_mask` in our code.
    `cond_mask`, for testing, it is `gt_mask`; for training, it is `observed_mask`
    includes some artificially missing values.

    """

    def __init__(
        self,
        data: Union[dict, str],
        target_strategy: str,
        return_X_ori: bool,
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        assert target_strategy in ["random", "hist", "mix"]
        self.target_strategy = target_strategy

    @staticmethod
    def get_rand_mask(observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(-1)
        sample_ratio = np.random.rand()  # missing ratio
        num_observed = observed_mask.sum().item()
        num_masked = round(num_observed * sample_ratio)
        rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask):
        cond_mask = observed_mask.clone()
        mask_choice = np.random.rand()
        if self.target_strategy == "mix" and mask_choice > 0.5:
            rand_mask = self.get_rand_mask(observed_mask)
            cond_mask = rand_mask
        else:
            cond_mask = cond_mask * for_pattern_mask
        return cond_mask

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx :
            The index to fetch the specified sample.

        Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            observed_data : tensor,
                Time-series data with all observed values for model input.

            indicating_mask : tensor,
                The mask records all artificially missing values to the model.

            cond_mask : tensor,
                The mask records all originally and artificially missing values to the model.

            observed_tp : tensor,
                The time points (timestamp) of the observed data.

        """

        if self.return_X_ori:
            observed_data = self.X_ori[idx]
            cond_mask = self.missing_mask[idx]
            indicating_mask = self.indicating_mask[idx]
        else:
            observed_data = self.X[idx]
            observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
            if self.target_strategy == "random":
                cond_mask = self.get_rand_mask(observed_mask)
            else:
                if "for_pattern_mask" in self.data.keys():
                    for_pattern_mask = torch.from_numpy(self.data["for_pattern_mask"][idx]).to(torch.float32)
                else:
                    previous_sample = self.X[idx - 1]
                    for_pattern_mask = (~torch.isnan(previous_sample)).to(torch.float32)

                cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
            indicating_mask = observed_mask - cond_mask

        observed_tp = (
            torch.arange(0, self.n_steps, dtype=torch.float32)
            if "time_points" not in self.data.keys()
            else torch.from_numpy(self.data["time_points"][idx]).to(torch.float32)
        )

        sample = [
            torch.tensor(idx),
            observed_data,
            indicating_mask,
            cond_mask,
            observed_tp,
        ]

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            observed_data : tensor,
                Time-series data with all observed values for model input.

            indicating_mask : tensor,
                The mask records all artificially missing values to the model.

            cond_mask : tensor,
                The mask records all originally and artificially missing values to the model.

            observed_tp : tensor,
                The time points (timestamp) of the observed data.

        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        if self.return_X_ori:
            observed_data = torch.from_numpy(self.file_handle["X_ori"][idx]).to(torch.float32)
            observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
            X = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            _, cond_mask = fill_and_get_mask_torch(X)
            indicating_mask = observed_mask - cond_mask
        else:
            observed_data = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
            observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
            if self.target_strategy == "random":
                cond_mask = self.get_rand_mask(observed_mask)
            else:
                if "for_pattern_mask" in self.data.keys():
                    for_pattern_mask = torch.from_numpy(self.file_handle["for_pattern_mask"][idx]).to(torch.float32)
                else:
                    previous_sample = torch.from_numpy(self.file_handle["X"][idx - 1]).to(torch.float32)
                    for_pattern_mask = (~torch.isnan(previous_sample)).to(torch.float32)

                cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
            indicating_mask = observed_mask - cond_mask

        observed_tp = (
            torch.arange(0, self.n_steps, dtype=torch.float32)
            if "time_points" not in self.file_handle.keys()
            else torch.from_numpy(self.file_handle["time_points"][idx]).to(torch.float32)
        )

        sample = [
            torch.tensor(idx),
            observed_data,
            indicating_mask,
            cond_mask,
            observed_tp,
        ]

        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample


class TestDatasetForCSDI(DatasetForCSDI):
    """Test dataset for CSDI model."""

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        file_type: str = "hdf5",
    ):
        super().__init__(data, "random", return_X_ori, file_type)

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx :
            The index to fetch the specified sample.

        Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            observed_data : tensor,
                Time-series data with all observed values for model input.

            cond_mask : tensor,
                The mask records missing values to the model.

            observed_tp : tensor,
                The time points (timestamp) of the observed data.
        """

        observed_data = self.X[idx]
        observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
        cond_mask = observed_mask

        observed_tp = (
            torch.arange(0, self.n_steps, dtype=torch.float32)
            if "time_points" not in self.data.keys()
            else torch.from_numpy(self.data["time_points"][idx]).to(torch.float32)
        )

        sample = [
            torch.tensor(idx),
            observed_data,
            cond_mask,
            observed_tp,
        ]

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx :
            The index of the sample to be return.

        Returns
        -------
        sample :
            A list contains

            index : int tensor,
                The index of the sample.

            observed_data : tensor,
                Time-series data with all observed values for model input.

            cond_mask : tensor,
                The mask records missing values to the model.

            observed_tp : tensor,
                The time points (timestamp) of the observed data.

        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        observed_data = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
        cond_mask = observed_mask

        observed_tp = (
            torch.arange(0, self.n_steps, dtype=torch.float32)
            if "time_points" not in self.file_handle.keys()
            else torch.from_numpy(self.file_handle["time_points"][idx]).to(torch.float32)
        )

        sample = [
            torch.tensor(idx),
            observed_data,
            cond_mask,
            observed_tp,
        ]

        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
