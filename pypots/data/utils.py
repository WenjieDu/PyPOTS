"""
Data utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import copy
from typing import Union

import benchpots
import numpy as np
import torch


def turn_data_into_specified_dtype(
    data: Union[np.ndarray, torch.Tensor, list],
    dtype: str = "tensor",
) -> Union[np.ndarray, torch.Tensor]:
    """Turn the given data into the specified data type."""

    if isinstance(data, torch.Tensor):
        data = data if dtype == "tensor" else data.numpy()
    elif isinstance(data, list):
        data = torch.tensor(data) if dtype == "tensor" else np.asarray(data)
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data) if dtype == "tensor" else data
    else:
        raise TypeError(
            f"data should be an instance of list/np.ndarray/torch.Tensor, but got {type(data)}"
        )
    return data


def _parse_delta_torch(missing_mask: torch.Tensor) -> torch.Tensor:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def cal_delta_for_single_sample(mask: torch.Tensor) -> torch.Tensor:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        # the first step in the delta matrix is all 0
        d = [torch.zeros(1, n_features, device=device)]

        for step in range(1, n_steps):
            d.append(
                torch.ones(1, n_features, device=device) + (1 - mask[step - 1]) * d[-1]
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


def _parse_delta_numpy(missing_mask: np.ndarray) -> np.ndarray:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing data (0 means missing values, 1 means observed values).

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def cal_delta_for_single_sample(mask: np.ndarray) -> np.ndarray:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        # the first step in the delta matrix is all 0
        d = [np.zeros(n_features)]

        for step in range(1, seq_len):
            d.append(np.ones(n_features) + (1 - mask[step - 1]) * d[-1])
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


def parse_delta(
    missing_mask: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Generate the time-gap matrix (i.e. the delta metrix) from the missing mask.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask :
        Binary masks indicate missing data (0 means missing values, 1 means observed values).
        Shape of [n_steps, n_features] or [n_samples, n_steps, n_features].

    Returns
    -------
    delta :
        The delta matrix indicates the time gaps between observed values.
        With the same shape of missing_mask.

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """
    if isinstance(missing_mask, np.ndarray):
        delta = _parse_delta_numpy(missing_mask)
    elif isinstance(missing_mask, torch.Tensor):
        delta = _parse_delta_torch(missing_mask)
    else:
        raise RuntimeError
    return delta


def sliding_window(
    time_series: Union[np.ndarray, torch.Tensor],
    window_len: int,
    sliding_len: int = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Generate time series samples with sliding window method, truncating windows from time-series data
    with a given sequence length.

    Given a time series of shape [seq_len, n_features] (seq_len is the total sequence length of the time series), this
    sliding_window function will generate time-series samples from this given time series with sliding window method.
    The number of generated samples is seq_len//sliding_len. And the final returned numpy ndarray has a shape
    [seq_len//sliding_len, n_steps, n_features].

    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]

    window_len :
        The length of the sliding window, i.e. the number of time steps in the generated data samples.

    sliding_len :
        The sliding length of the window for each moving step. It will be set as the same with n_steps if None.

    Returns
    -------
    samples :
        The generated time-series data samples of shape [seq_len//sliding_len, n_steps, n_features].

    """

    return benchpots.utils.sliding_window(
        time_series,
        window_len,
        sliding_len,
    )


def inverse_sliding_window(X, sliding_len):
    """Restore the original time-series data from the generated sliding window samples.
    Note that this is the inverse operation of the `sliding_window` function, but there is no guarantee that
    the restored data is the same as the original data considering that
    1. the sliding length may be larger than the window size and there will be gaps between restored data;
    2. if values in the samples get changed, the overlap part may not be the same as the original data after averaging;
    3. some incomplete samples at the tail may be dropped during the sliding window operation, hence the restored data
       may be shorter than the original data.

    Parameters
    ----------
    X :
        The generated time-series samples with sliding window method, shape of [n_samples, n_steps, n_features],
        where n_steps is the window size of the used sliding window method.

    sliding_len :
        The sliding length of the window for each moving step in the sliding window method used to generate X.

    Returns
    -------
    restored_data :
        The restored time-series data with shape of [total_length, n_features].

    """

    return benchpots.utils.inverse_sliding_window(
        X,
        sliding_len,
    )


def parse_delta_bidirectional(masks, direction):
    if direction == 'backward':
        masks = masks[::-1] == 1.0

    [T, D] = masks.shape
    deltas = []
    for t in range(T):
        if t == 0:
            deltas.append(np.ones(D))
        else:
            deltas.append(np.ones(D) + (1 - masks[t]) * deltas[-1])

    return np.array(deltas)

def compute_last_obs(data, masks, direction):
    """
    Compute the last observed values for each time step.
    
    Parameters:
    - data (np.array): Original data array.
    - masks (np.array): Binary masks indicating where data is not NaN.

    Returns:
    - last_obs (np.array): Array of the last observed values.
    """
    if direction == 'backward':
        masks = masks[::-1] == 1.0
        data = data[::-1]
    
    [T, D] = masks.shape
    last_obs = np.full((T, D), np.nan)  # Initialize last observed values with NaNs
    last_obs_val = np.full(D, np.nan)  # Initialize last observed values for first time step with NaNs
    
    for t in range(1, T):
        mask = masks[t-1]
        # Update last observed values
        # last_obs_val = np.where(masks[t], data[t], last_obs_val) 
        last_obs_val[mask] = data[t-1, mask]
        # Assign last observed values to the current time step
        last_obs[t] = last_obs_val 
    
    return last_obs

def adjust_probability_vectorized(obs_count, avg_count, base_prob, increase_factor=0.5):
    if obs_count < avg_count:
        return min(base_prob * (avg_count / obs_count) * increase_factor, 1.0)
    else:
        return max(base_prob * (obs_count / avg_count) / increase_factor, 0)



def normalize_csai(data, mean, std, compute_intervals=False):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]


    measure = copy.deepcopy(data).reshape(n_patients * n_hours, n_variables)

    if compute_intervals:
        intervals_list = {v: [] for v in range(n_variables)}

    isnew = 0
    if len(mean) == 0 or len(std) == 0:
        isnew = 1
        mean_set = np.zeros([n_variables])
        std_set = np.zeros([n_variables])
    else:
        mean_set = mean
        std_set = std
        
    for v in range(n_variables):

        if isnew:
            mask_v = ~np.isnan(measure[:,v]) * 1
            idx_global = np.where(mask_v == 1)[0]

            if idx_global.sum() == 0:
                continue

            measure_mean = np.mean(measure[:, v][idx_global])
            measure_std = np.std(measure[:, v][idx_global])

            mean_set[v] = measure_mean
            std_set[v] = measure_std
        else:
            measure_mean = mean[v]
            measure_std = std[v]
        for p in range(n_patients):
            mask_p_v = ~np.isnan(data[p, :, v]) * 1
            idx_p = np.where(mask_p_v == 1)[0]
            if compute_intervals and len(idx_p) > 1:
                intervals_list[v].extend([idx_p[i+1] - idx_p[i] for i in range(len(idx_p)-1)])

            for ix in idx_p:
                if measure_std != 0:
                    data[p, ix, v] = (data[p, ix, v] - measure_mean) / measure_std
                else:
                    data[p, ix, v] = data[p, ix, v] - measure_mean

    if compute_intervals:
        intervals_list = {v: np.median(intervals_list[v]) if intervals_list[v] else np.nan for v in intervals_list}

    if compute_intervals:
        return data, mean_set, std_set, intervals_list
    else:
        return data, mean_set, std_set



def non_uniform_sample_loader_bidirectional(data, removal_percent, pre_replacement_probabilities=None, increase_factor=0.5):

    # Random seed
    np.random.seed(1)
    torch.manual_seed(1)

    # Get Dimensionality
    [N, T, D] = data.shape

    if pre_replacement_probabilities is None:


        observations_per_feature = np.sum(~np.isnan(data), axis=(0, 1))
        average_observations = np.mean(observations_per_feature)
        replacement_probabilities = np.full(D, removal_percent / 100)

        if increase_factor > 0:
            print('The increase_factor is {}!'.format(increase_factor))
            for feature_idx in range(D):
                replacement_probabilities[feature_idx] = adjust_probability_vectorized(
                    observations_per_feature[feature_idx],
                    average_observations,
                    replacement_probabilities[feature_idx],
                    increase_factor=increase_factor
                )
            
            # print('before:\n',replacement_probabilities)
            total_observations = np.sum(observations_per_feature)
            total_replacement_target = total_observations * removal_percent / 100

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = np.sum(replacement_probabilities * observations_per_feature)
                if np.isclose(total_replacement, total_replacement_target, rtol=1e-3):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
            
            # print('after:\n',replacement_probabilities)
    else:
        replacement_probabilities = pre_replacement_probabilities

    recs = []
    number = 0
    masks_sum = np.zeros(D)
    eval_masks_sum = np.zeros(D)
    values = copy.deepcopy(data)
    random_matrix = np.random.rand(N, T, D)
    values[(~np.isnan(values)) & (random_matrix < replacement_probabilities)] = np.nan
#     mask = (~torch.isnan(values)) & (random_matrix < replacement_probabilities)

# # Ensure mask is boolean
#     mask = mask.to(torch.bool)

#     # Apply mask
#     values[mask] = float('nan')
    for i in range(N):
        masks = ~np.isnan(values[i, :, :])
        eval_masks = (~np.isnan(values[i, :, :])) ^ (~np.isnan(data[i, :, :]))
        evals = data[i, :, :]
        rec = {}
        # rec['label'] = label[i]
        deltas_f = parse_delta_bidirectional(masks, direction='forward')
        deltas_b = parse_delta_bidirectional(masks, direction='backward')
        last_obs_f = compute_last_obs(values[i, :, :], masks, direction='forward')
        last_obs_b = compute_last_obs(values[i, :, :], masks, direction='backward')
        rec['values'] = np.nan_to_num(values[i, :, :]).tolist()
        rec['last_obs_f'] = np.nan_to_num(last_obs_f).tolist()
        rec['last_obs_b'] = np.nan_to_num(last_obs_b).tolist()
        rec['masks'] = masks.astype('int32').tolist()
        rec['evals'] = np.nan_to_num(evals).tolist()
        rec['eval_masks'] = eval_masks.astype('int32').tolist()
        rec['deltas_f'] = deltas_f.tolist()
        rec['deltas_b'] = deltas_b.tolist()
        recs.append(rec)
        number += 1
        masks_sum += np.sum(masks, axis=0)
        eval_masks_sum += np.sum(eval_masks, axis=0)

    
    return recs, replacement_probabilities


def collate_fn_bidirectional(recs):

    def to_tensor_dict(recs):

        values = torch.FloatTensor(np.array([r['values'] for r in recs]))
        last_obs_f = torch.FloatTensor(np.array([r['last_obs_f'] for r in recs]))
        last_obs_b = torch.FloatTensor(np.array([r['last_obs_b'] for r in recs]))
        masks = torch.FloatTensor(np.array([r['masks'] for r in recs]))
        deltas_f = torch.FloatTensor(np.array([r['deltas_f'] for r in recs]))
        deltas_b = torch.FloatTensor(np.array([r['deltas_b'] for r in recs]))

        evals = torch.FloatTensor(np.array([r['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r['eval_masks'] for r in recs]))

        return {'values': values,
                'last_obs_f': last_obs_f,
                'last_obs_b': last_obs_b,
                'masks': masks,
                'deltas_f': deltas_f,
                'deltas_b': deltas_b,
                'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = to_tensor_dict(recs)

    # ret_dict['labels'] = torch.FloatTensor(np.array([r['label'] for r in recs]))
    # ret_dict['labels'] = torch.LongTensor(np.array([r['label'] for r in recs]))

    return ret_dict