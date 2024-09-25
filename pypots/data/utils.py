"""
Data utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

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
        raise TypeError(f"data should be an instance of list/np.ndarray/torch.Tensor, but got {type(data)}")
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
            d.append(torch.ones(1, n_features, device=device) + (1 - mask[step - 1]) * d[-1])
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


def parse_delta(missing_mask: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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
