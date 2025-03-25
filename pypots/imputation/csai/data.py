"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import copy
from typing import Iterable, Union

import numpy as np
import torch
from pygrinder import mnar_nonuniform

from ...data.dataset.base import BaseDataset
from ...data.utils import parse_delta


def compute_intervals(data):
    """
    Compute the time intervals between observations for each variable in the dataset.

    Parameters
    ----------
    data : np.ndarray
        The input time-series data of shape [n_patients, n_hours, n_variables], which may contain missing values (NaNs).

    Returns
    -------
    intervals_list : dict of int to float
        A dictionary containing the median time intervals between observations for each variable.
    """
    n_patients, _, n_variables = data.shape
    intervals_list = {}

    for v in range(n_variables):
        all_intervals = []
        # Loop over each patient
        for p in range(n_patients):
            # Get non-NaN observation indices for the current patient and variable
            valid_time_points = np.where(~np.isnan(data[p, :, v]))[0]

            # If the patient has more than one valid observation, compute time intervals
            if len(valid_time_points) > 1:
                # Calculate time differences between consecutive observations
                intervals = np.diff(valid_time_points)
                all_intervals.extend(intervals)

        # Compute the median interval for the current variable
        intervals_list[v] = np.median(all_intervals) if all_intervals else np.nan

    return intervals_list


def compute_last_obs(data, masks):
    """
    Compute the last observed values for each time step.

    Parameters:
    - data (np.array): Original data array of shape [T, D].
    - masks (np.array): Binary masks indicating where data is not NaN, of shape [T, D].

    Returns:
    - last_obs (np.array): Array of the last observed values, of shape [T, D].
    """
    T, D = masks.shape
    last_obs = np.full((T, D), np.nan)  # Initialize last observed values with NaNs
    last_obs_val = np.full(D, np.nan)  # Initialize last observed values for first time step with NaNs

    for t in range(1, T):  # Start from t=1, keeping first row as NaN
        mask = masks[t - 1]
        # Update last observed values based on previous time step
        last_obs_val[mask] = data[t - 1, mask]
        # Assign last observed values to the current time step
        last_obs[t] = last_obs_val

    return last_obs


def non_uniform_sample(data, removal_percent, pre_replacement_probabilities=None, increase_factor=0.5):
    """
    Process time-series data by randomly removing a certain percentage of observed values based on pre-defined
    replacement probabilities, and compute the necessary features such as forward and backward deltas, masks,
    and last observed values.

    This function generates records for each time series and returns them as PyTorch tensors for further usage.

    Parameters
    ----------
    data : np.ndarray
        The input data with shape [N, T, D], where N is the number of samples, T is the number of time steps,
        and D is the number of features. Missing values should be indicated with NaNs.

    removal_percent : float
        The percentage of observed values to be removed randomly from the dataset.

    pre_replacement_probabilities : np.ndarray, optional
        Pre-defined replacement probabilities for each feature. If provided, this will be used to determine
        which values to remove.

    increase_factor : float, default=0.5
        A factor to adjust replacement probabilities based on the observation count for each feature.

    Returns
    -------
    tensor_dict : dict of torch.Tensors
        A dictionary of PyTorch tensors including 'values', 'last_obs_f', 'last_obs_b', 'masks', 'deltas_f',
        'deltas_b', 'evals', and 'eval_masks'.

    replacement_probabilities : np.ndarray
        The computed or provided replacement probabilities for each feature.
    """
    # Prepare data structures
    recs = []
    values = copy.deepcopy(data)

    # Generate missing values based on replacement probabilities
    values, replacement_probabilities = mnar_nonuniform(
        values, removal_percent, pre_replacement_probabilities, increase_factor
    )

    # Generate records and features for each sample
    for i in range(data.shape[0]):
        masks = ~np.isnan(values[i, :, :])
        eval_masks = (~np.isnan(values[i, :, :])) ^ (~np.isnan(data[i, :, :]))
        evals = data[i, :, :]

        # Compute forward and backward deltas
        deltas_f = parse_delta(masks)
        deltas_b = parse_delta(masks[::-1, :])

        # Compute last observations for forward and backward directions
        last_obs_f = compute_last_obs(values[i, :, :], masks)
        last_obs_b = compute_last_obs(values[i, ::-1, :], masks[::-1, :])

        # Append the record for this sample
        recs.append(
            {
                "values": np.nan_to_num(values[i, :, :]),
                "last_obs_f": np.nan_to_num(last_obs_f),
                "last_obs_b": np.nan_to_num(last_obs_b),
                "masks": masks.astype("int32"),
                "evals": np.nan_to_num(evals),
                "eval_masks": eval_masks.astype("int32"),
                "deltas_f": deltas_f,
                "deltas_b": deltas_b,
            }
        )

    # Convert records to PyTorch tensors
    tensor_dict = {
        "values": torch.FloatTensor(np.array([r["values"] for r in recs])),
        "last_obs_f": torch.FloatTensor(np.array([r["last_obs_f"] for r in recs])),
        "last_obs_b": torch.FloatTensor(np.array([r["last_obs_b"] for r in recs])),
        "masks": torch.FloatTensor(np.array([r["masks"] for r in recs])),
        "deltas_f": torch.FloatTensor(np.array([r["deltas_f"] for r in recs])),
        "deltas_b": torch.FloatTensor(np.array([r["deltas_b"] for r in recs])),
        "evals": torch.FloatTensor(np.array([r["evals"] for r in recs])),
        "eval_masks": torch.FloatTensor(np.array([r["eval_masks"] for r in recs])),
    }

    return tensor_dict, replacement_probabilities


class DatasetForCSAI(BaseDataset):
    """ "
    Parameters
    ----------
    data :
        The dataset for model input, which can be either a dictionary or a path string to a data file.
        If it's a dictionary, `X` should be an array-like structure
        with shape [n_samples, n_steps, n_features], containing the time-series data,
        and it can have missing values. Optionally, the dictionary can include `y`,
        an array-like structure with shape [n_samples], representing the labels of `X`.
        If `data` is a path string, it should point to a data file (e.g., h5 file) that contains key-value pairs like
        a dictionary, including keys for `X` and possibly `y`.

    return_X_ori :
        Whether to return the original time-series data (`X_ori`) when fetching data samples,
        useful for evaluation purposes.

    return_y :
        Whether to return classification labels in the `__getitem__()` method if they exist in the dataset.
        If `True`, labels will be included in the returned data samples,
        which is useful for training classification models.
        If `False`, the labels won't be returned, suitable for testing or validation stages.

    file_type :
        The type of the data file if `data` is a path string, such as "hdf5".

    removal_percent :
        The percentage of data to be removed for simulating missing values during training.

    increase_factor :
        A scaling factor to increase the probability of missing data during training.

    replacement_probabilities :
        Optional precomputed probabilities for sampling missing values.
        If not provided, they will be calculated during the initialization of the dataset.

    Notes
    -----
    The DatasetForCSAI class is designed for bidirectional imputation of time-series data,
    handling both forward and backward directions to improve imputation accuracy.
    It supports on-the-fly data normalization and missing value simulation,
    making it suitable for training and evaluating deep learning models like CSAI.
    The class can work with large datasets stored on disk, leveraging lazy-loading to minimize memory usage,
    and supports both training and testing scenarios, adjusting data handling as needed.

    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        removal_percent: float = 0.0,
        increase_factor: float = 0.1,
        replacement_probabilities=None,
    ):
        super().__init__(
            data=data, return_X_ori=return_X_ori, return_X_pred=False, return_y=return_y, file_type=file_type
        )

        self.removal_percent = removal_percent
        self.increase_factor = increase_factor

        if not isinstance(self.data, str):
            self.intervals = compute_intervals(self.data["X"])

            if replacement_probabilities is None:
                self.processed_data, self.replacement_probabilities = non_uniform_sample(
                    data=self.data["X"],
                    removal_percent=self.removal_percent,
                    increase_factor=self.increase_factor,
                )
            else:
                self.replacement_probabilities = replacement_probabilities
                self.processed_data, _ = non_uniform_sample(
                    data=self.data["X"],
                    removal_percent=self.removal_percent,
                    pre_replacement_probabilities=self.replacement_probabilities,
                    increase_factor=self.increase_factor,
                )

            self.forward_X = self.processed_data["values"]
            self.forward_missing_mask = self.processed_data["masks"]
            self.backward_X = torch.flip(self.forward_X, dims=[1])
            self.backward_missing_mask = torch.flip(self.forward_missing_mask, dims=[1])

            self.X_ori = self.processed_data["evals"]
            self.indicating_mask = self.processed_data["eval_masks"]

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data from self.X if it is given.

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

            X : tensor,
                The feature vector for model input.

            missing_mask : tensor,
                The mask indicates all missing values in X.

            delta : tensor,
                The delta matrix contains time gaps of missing values.

            label (optional) : tensor,
                The target label of the time-series sample.

            last_obs : tensor,
                The last observed values for each time step.

            replacement_probabilities : np.ndarray,
                The computed or provided replacement probabilities for each feature.

            intervals : dict of int to float,
                A dictionary containing the median time intervals between observations for each variable.

        """

        sample = [
            torch.tensor(idx),
            # for forward
            self.forward_X[idx],
            self.forward_missing_mask[idx],
            self.processed_data["deltas_f"][idx],
            self.processed_data["last_obs_f"][idx],
            # for backward
            self.backward_X[idx],
            self.backward_missing_mask[idx],
            self.processed_data["deltas_b"][idx],
            self.processed_data["last_obs_b"][idx],
        ]

        if self.return_X_ori:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return {
            "sample": sample,
            "replacement_probabilities": self.replacement_probabilities,
            "intervals": self.intervals,
        }

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        raise NotImplementedError(
            "CSAI does not support lazy loading because normalise mean and std need to be calculated ahead."
        )
