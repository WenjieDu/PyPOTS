"""
Dataset class for the forecasting model CSDI.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Iterable

import numpy as np
import torch
from pygrinder import fill_and_get_mask_torch

from ...data.dataset.base import BaseDataset


class DatasetForCSDI(BaseDataset):
    """Dataset for CSDI forecasting model."""

    def __init__(
        self,
        data: Union[dict, str],
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            return_X_ori=False,
            return_X_pred=True,
            return_y=False,
            file_type=file_type,
        )

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        ind = np.arange(self.n_pred_features)
        np.random.shuffle(ind)

        extracted_data = observed_data[:, ind[: self.n_features]]
        extracted_mask = observed_mask[:, ind[: self.n_features]]
        extracted_feature_id = feature_id[ind[: self.n_features]]
        extracted_gt_mask = gt_mask[:, ind[: self.n_features]]

        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask

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

        feature_id = torch.arange(self.n_pred_features)
        observed_data = self.X[idx]
        observed_data, observed_mask = fill_and_get_mask_torch(observed_data)

        # apply specifically given mask or the hist masking strategy, rather than the random masking strategy
        if "for_pattern_mask" in self.data.keys():
            for_pattern_mask = torch.from_numpy(self.data["for_pattern_mask"][idx]).to(torch.float32)
        else:
            previous_sample = self.X[idx - 1]
            for_pattern_mask = (~torch.isnan(previous_sample)).to(torch.float32)
        cond_mask = observed_mask * for_pattern_mask

        indicating_mask = observed_mask - cond_mask

        if self.n_pred_features > self.n_features:
            (
                observed_data,
                observed_mask,
                feature_id,
                cond_mask,
            ) = self.sample_features(observed_data, observed_mask, feature_id, cond_mask)

        X_pred = self.X_pred[idx]
        X_pred_missing_mask = self.X_pred_missing_mask[idx]

        observed_data = torch.concat([observed_data, X_pred], dim=0)
        indicating_mask = torch.concat([indicating_mask, X_pred_missing_mask], dim=0)
        cond_mask = torch.concat([cond_mask, torch.zeros(X_pred.shape)], dim=0)
        observed_tp = torch.arange(0, self.n_steps + self.n_pred_steps, dtype=torch.float32)

        sample = [
            torch.tensor(idx),
            observed_data,
            indicating_mask,
            cond_mask,
            observed_tp,
            feature_id,
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

        feature_id = torch.arange(self.n_pred_features)
        observed_data = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        observed_data, observed_mask = fill_and_get_mask_torch(observed_data)

        # apply specifically given mask or the hist masking strategy, rather than the random masking strategy
        if "for_pattern_mask" in self.file_handle.keys():
            for_pattern_mask = torch.from_numpy(self.file_handle["for_pattern_mask"][idx]).to(torch.float32)
        else:
            previous_sample = torch.from_numpy(self.file_handle["X"][idx - 1]).to(torch.float32)
            for_pattern_mask = (~torch.isnan(previous_sample)).to(torch.float32)
        cond_mask = observed_mask * for_pattern_mask

        indicating_mask = observed_mask - cond_mask

        if self.n_pred_features > self.n_features:
            (
                observed_data,
                observed_mask,
                feature_id,
                cond_mask,
            ) = self.sample_features(observed_data, observed_mask, feature_id, cond_mask)

        X_pred = torch.from_numpy(self.file_handle["X_pred"][idx]).to(torch.float32)
        X_pred, X_pred_missing_mask = fill_and_get_mask_torch(X_pred)

        observed_data = torch.concat([observed_data, X_pred], dim=0)
        indicating_mask = torch.concat([indicating_mask, X_pred_missing_mask], dim=0)
        cond_mask = torch.concat([cond_mask, torch.zeros(X_pred.shape)], dim=0)
        observed_tp = torch.arange(0, self.n_steps + self.n_pred_steps, dtype=torch.float32)

        sample = [
            torch.tensor(idx),
            observed_data,
            indicating_mask,
            cond_mask,
            observed_tp,
            feature_id,
        ]

        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample


class TestDatasetForCSDI(DatasetForCSDI):
    """Test dataset for CSDI forecasting model."""

    def __init__(
        self,
        data: Union[dict, str],
        n_pred_steps: int,
        n_pred_features: int,
        file_type: str = "hdf5",
    ):
        super().__init__(
            data=data,
            file_type=file_type,
        )
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

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

        feature_id = torch.arange(self.n_pred_features)
        observed_data = self.X[idx]
        observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
        cond_mask = observed_mask

        if self.n_pred_features > self.n_features:
            (
                observed_data,
                observed_mask,
                feature_id,
                cond_mask,
            ) = self.sample_features(observed_data, observed_mask, feature_id, cond_mask)

        observed_data = torch.concat(
            [observed_data, torch.zeros([self.n_pred_steps, self.n_pred_features])],
            dim=0,
        )

        cond_mask = torch.concat([cond_mask, torch.zeros([self.n_pred_steps, self.n_pred_features])], dim=0)
        observed_tp = torch.arange(0, self.n_steps + self.n_pred_steps, dtype=torch.float32)

        sample = [
            torch.tensor(idx),
            observed_data,
            cond_mask,
            observed_tp,
            feature_id,
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

        feature_id = torch.arange(self.n_pred_features)
        observed_data = torch.from_numpy(self.file_handle["X"][idx]).to(torch.float32)
        observed_data, observed_mask = fill_and_get_mask_torch(observed_data)
        cond_mask = observed_mask

        if self.n_pred_features > self.n_features:
            (
                observed_data,
                observed_mask,
                feature_id,
                cond_mask,
            ) = self.sample_features(observed_data, observed_mask, feature_id, cond_mask)

        observed_data = torch.concat(
            [observed_data, torch.zeros([self.n_pred_steps, self.n_pred_features])],
            dim=0,
        )

        cond_mask = torch.concat([cond_mask, torch.zeros([self.n_pred_steps, self.n_pred_features])], dim=0)
        observed_tp = torch.arange(0, self.n_steps + self.n_pred_steps, dtype=torch.float32)

        feature_id = torch.arange(self.n_pred_features)

        sample = [
            torch.tensor(idx),
            observed_data,
            cond_mask,
            observed_tp,
            feature_id,
        ]

        if self.return_y:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample
