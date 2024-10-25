"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

from typing import Iterable
from ...data.dataset import BaseDataset
import numpy as np
import torch
from typing import Union
import copy
from ...data.utils import parse_delta
from sklearn.preprocessing import StandardScaler


def normalize_csai(
    data,
    mean: list = None,
    std: list = None,
    compute_intervals: bool = False,
):
    """
    Normalize the data based on the given mean and standard deviation, and optionally compute time intervals between observations.

    Parameters
    ----------
    data : np.ndarray
        The input time-series data of shape [n_patients, n_hours, n_variables], which may contain missing values (NaNs).

    mean : list of float, optional
        The mean values for each variable, used for normalization. If empty, means will be computed from the data.

    std : list of float, optional
        The standard deviation values for each variable, used for normalization. If empty, std values will be computed from the data.

    compute_intervals : bool, optional, default=False
        Whether to compute the time intervals between observations for each variable.

    Returns
    -------
    data : torch.Tensor
        The normalized time-series data with the same shape as the input data, moved to the specified device.

    mean_set : np.ndarray
        The mean values for each variable after normalization, either computed from the data or passed as input.

    std_set : np.ndarray
        The standard deviation values for each variable after normalization, either computed from the data or passed as input.

    intervals_list : dict of int to float, optional
        If `compute_intervals` is True, this will return the median time intervals between observations for each variable.
    """

    # Convert data to numpy array if it is a torch tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    n_patients, n_hours, n_variables = data.shape

    # Flatten data for easier computation of statistics
    reshaped_data = data.reshape(-1, n_variables)

    # Use StandardScaler for normalization
    scaler = StandardScaler()

    # Update condition to handle empty list as well
    if mean is None or std is None or len(mean) == 0 or len(std) == 0:
        # Fit the scaler on the data (ignores NaNs during the fitting process)
        scaler.fit(reshaped_data)
        mean_set = scaler.mean_
        std_set = scaler.scale_
    else:
        # Use provided mean and std by directly setting them in the scaler
        scaler.mean_ = np.array(mean)
        scaler.scale_ = np.array(std)
        mean_set = np.array(mean)
        std_set = np.array(std)

    # Transform data using scaler, which ignores NaNs
    scaled_data = scaler.transform(reshaped_data)

    # Reshape back to original shape [n_patients, n_hours, n_variables]
    normalized_data = scaled_data.reshape(n_patients, n_hours, n_variables)

    # Optimized interval calculation considering NaNs in each patient
    if compute_intervals:
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
    else:
        intervals_list = None

    return normalized_data, mean_set, std_set, intervals_list


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


def adjust_probability_vectorized(
    obs_count: Union[int, float], avg_count: Union[int, float], base_prob: float, increase_factor: float = 0.5
) -> float:
    """
    Adjusts the base probability based on observed and average counts using a scaling factor.

    Parameters
    ----------
    obs_count : int or float
        The observed count of an event or observation in the dataset.

    avg_count : int or float
        The average count of the event or observation across the dataset.

    base_prob : float
        The base probability of the event or observation occurring.

    increase_factor : float, optional, default=0.5
        A scaling factor applied to adjust the probability when `obs_count` is below `avg_count`.
        This factor influences how much to increase or decrease the probability.

    Returns
    -------
    float
        The adjusted probability, scaled based on the ratio between the observed count and the average count.
        The adjusted probability will be within the range [0, 1].

    Notes
    -----
    This function adjusts a base probability based on the observed count (`obs_count`) compared to the average count
    (`avg_count`). If the observed count is lower than the average, the probability is increased proportionally,
    but capped at a maximum of 1.0. Conversely, if the observed count exceeds the average, the probability is reduced,
    but not below 0. The `increase_factor` controls the sensitivity of the probability adjustment when the observed
    count is less than the average count.
    """
    if obs_count < avg_count:
        # Increase probability when observed count is lower than average count
        return min(base_prob * (avg_count / obs_count) * increase_factor, 1.0)
    else:
        # Decrease probability when observed count exceeds average count
        return max(base_prob * (obs_count / avg_count) / increase_factor, 0.0)


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
    # Get dimensionality
    [N, T, D] = data.shape

    # Compute replacement probabilities if not provided
    if pre_replacement_probabilities is None:
        observations_per_feature = np.sum(~np.isnan(data), axis=(0, 1))
        average_observations = np.mean(observations_per_feature)
        replacement_probabilities = np.full(D, removal_percent / 100)

        if increase_factor > 0:
            for feature_idx in range(D):
                replacement_probabilities[feature_idx] = adjust_probability_vectorized(
                    observations_per_feature[feature_idx],
                    average_observations,
                    replacement_probabilities[feature_idx],
                    increase_factor=increase_factor,
                )

            total_observations = np.sum(observations_per_feature)
            total_replacement_target = total_observations * removal_percent / 100

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = np.sum(replacement_probabilities * observations_per_feature)
                if np.isclose(total_replacement, total_replacement_target, rtol=1e-3):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
    else:
        replacement_probabilities = pre_replacement_probabilities

    # Prepare data structures
    recs = []
    values = copy.deepcopy(data)

    # Randomly remove data points based on replacement probabilities
    random_matrix = np.random.rand(N, T, D)
    values[(~np.isnan(values)) & (random_matrix < replacement_probabilities)] = np.nan

    # Generate records and features for each sample
    for i in range(N):
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
        The dataset for model input, which can be either a dictionary or a path string to a data file. If it's a dictionary, `X` should be an array-like structure with shape [n_samples, sequence length (n_steps), n_features], containing the time-series data, and it can have missing values. Optionally, the dictionary can include `y`, an array-like structure with shape [n_samples], representing the labels of `X`. If `data` is a path string, it should point to a data file (e.g., h5 file) that contains key-value pairs like a dictionary, including keys for `X` and possibly `y`.

    return_X_ori :
        Whether to return the original time-series data (`X_ori`) when fetching data samples, useful for evaluation purposes.

    return_y :
        Whether to return classification labels in the `__getitem__()` method if they exist in the dataset. If `True`, labels will be included in the returned data samples, which is useful for training classification models. If `False`, the labels won't be returned, suitable for testing or validation stages.

    file_type :
        The type of the data file if `data` is a path string, such as "hdf5".

    removal_percent :
        The percentage of data to be removed for simulating missing values during training.

    increase_factor :
        A scaling factor to increase the probability of missing data during training.

    compute_intervals :
        Whether to compute time intervals between observations for handling irregular time-series data.

    replacement_probabilities :
        Optional precomputed probabilities for sampling missing values. If not provided, they will be calculated during the initialization of the dataset.

    normalise_mean :
        A list of mean values for normalizing the input features. If not provided, they will be computed during initialization.

    normalise_std :
        A list of standard deviation values for normalizing the input features. If not provided, they will be computed during initialization.

    training :
        Whether the dataset is used for training. If `False`, it will adjust how data is processed, particularly for evaluation and testing phases.

    Notes
    -----
    The DatasetForCSAI class is designed for bidirectional imputation of time-series data, handling both forward and backward directions to improve imputation accuracy. It supports on-the-fly data normalization and missing value simulation, making it suitable for training and evaluating deep learning models like CSAI. The class can work with large datasets stored on disk, leveraging lazy-loading to minimize memory usage, and supports both training and testing scenarios, adjusting data handling as needed.

    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        removal_percent: float = 0.0,
        increase_factor: float = 0.1,
        compute_intervals: bool = False,
        replacement_probabilities=None,
        normalise_mean: list = [],
        normalise_std: list = [],
        training: bool = True,
    ):
        super().__init__(
            data=data, return_X_ori=return_X_ori, return_X_pred=False, return_y=return_y, file_type=file_type
        )

        self.removal_percent = removal_percent
        self.increase_factor = increase_factor
        self.compute_intervals = compute_intervals
        self.replacement_probabilities = replacement_probabilities
        self.normalise_mean = normalise_mean
        self.normalise_std = normalise_std
        self.training = training

        if not isinstance(self.data, str):
            self.normalized_data, self.mean_set, self.std_set, self.intervals = normalize_csai(
                self.data["X"],
                self.normalise_mean,
                self.normalise_std,
                compute_intervals,
            )

            self.processed_data, self.replacement_probabilities = non_uniform_sample(
                self.normalized_data,
                removal_percent,
                replacement_probabilities,
                increase_factor,
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

        if not self.training:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return {
            "sample": sample,
            "replacement_probabilities": self.replacement_probabilities,
            "mean_set": self.mean_set,
            "std_set": self.std_set,
            "intervals": self.intervals,
        }

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
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx])
        normalized_data, mean_set, std_set, intervals = normalize_csai(
            X,
            self.normalise_mean,
            self.normalise_std,
            self.compute_intervals,
        )

        processed_data, replacement_probabilities = non_uniform_sample(
            normalized_data,
            self.removal_percent,
            self.replacement_probabilities,
            self.increase_factor,
        )
        forward_X = processed_data["values"]
        forward_missing_mask = processed_data["masks"]
        backward_X = torch.flip(forward_X, dims=[1])
        backward_missing_mask = torch.flip(forward_missing_mask, dims=[1])

        X_ori = self.processed_data["evals"]
        indicating_mask = self.processed_data["eval_masks"]

        if self.return_y:
            y = self.processed_data["labels"]

        sample = [
            torch.tensor(idx),
            # for forward
            forward_X,
            forward_missing_mask,
            processed_data["deltas_f"],
            processed_data["last_obs_f"],
            # for backward
            backward_X,
            backward_missing_mask,
            processed_data["deltas_b"],
            processed_data["last_obs_b"],
        ]

        if self.return_X_ori:
            sample.extend([X_ori, indicating_mask])

        # if the dataset has labels and is for training, then fetch it from the file
        if self.return_y:
            sample.append(y)

        return {
            "sample": sample,
            "replacement_probabilities": replacement_probabilities,
            "mean_set": mean_set,
            "std_set": std_set,
            "intervals": intervals,
        }
