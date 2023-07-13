"""
Data utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Union

import numpy as np
import pycorruptor as corruptor
import torch
from tsdb import (
    pickle_load as _pickle_load,
    pickle_dump as _pickle_dump,
)

pickle_load = _pickle_load
pickle_dump = _pickle_dump


def cal_missing_rate(X: Union[np.ndarray, torch.Tensor, list]) -> float:
    """Calculate the missing rate of the given data.

    Parameters
    ----------
    X :
        The data to calculate missing rate.

    Returns
    -------
    missing_rate :
        The missing rate of the given data.

    """
    missing_rate = corruptor.cal_missing_rate(X)
    return missing_rate


def masked_fill(
    X: Union[np.ndarray, torch.Tensor, list],
    mask: Union[np.ndarray, torch.Tensor, list],
    value: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Fill the masked values in ``X`` according to ``mask`` with the given ``value``.

    Parameters
    ----------
    X :
        The data to be filled.

    mask :
        The mask for filling the given data.

    value :
        The value to fill the masked values.

    Returns
    -------
    filled_X :
        The filled data.

    """
    filled_X = corruptor.masked_fill(X, mask, value)
    return filled_X


def mcar(
    X: Union[np.ndarray, torch.Tensor, list],
    rate: float,
    nan: float = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """Generate missing values in the given data with MCAR (Missing Completely At Random) mechanism.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    rate :
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.

        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.

    nan :
        Value used to fill NaN values.

    Returns
    -------
    X_intact : array,
        Original data with missing values (nan) filled with given parameter `nan`, with observed values intact.
        X_intact is for loss calculation in the masked imputation task.

    X : array,
        Original X with artificial missing values. X is for model input.
        Both originally-missing and artificially-missing values are filled with given parameter `nan`.

    missing_mask : array,
        The mask indicates all missing values in X.
        In it, 1 indicates observed values, and 0 indicates missing values.

    indicating_mask : array,
        The mask indicates the artificially-missing values in X, namely missing parts different from X_intact.
        In it, 1 indicates artificially missing values, and other values are indicated as 0.
    """
    X = corruptor.mcar(X, rate, nan)
    return X


def torch_parse_delta(missing_mask: torch.Tensor) -> torch.Tensor:
    """Generate time-gap (delta) matrix from missing masks.
    Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask :
        Binary masks indicate missing values. Shape of [n_steps, n_features] or [n_samples, n_steps, n_features]

    Returns
    -------
    delta
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask: torch.Tensor) -> torch.Tensor:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
        d = []
        for step in range(n_steps):
            if step == 0:
                d.append(torch.zeros(1, n_features, device=device))
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


def numpy_parse_delta(missing_mask: np.ndarray) -> np.ndarray:
    """Generate time-gap (delta) matrix from missing masks. Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask :
        Binary masks indicate missing values. Shape of [n_steps, n_features] or [n_samples, n_steps, n_features].

    Returns
    -------
    delta
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask: np.ndarray) -> np.ndarray:
        """calculate single sample's delta. The sample's shape is [n_steps, n_features]."""
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
