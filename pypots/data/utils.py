"""
Data utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import numpy as np
import torch


import pycorruptor as corruptor
from tsdb import (
    pickle_load as _pickle_load,
    pickle_dump as _pickle_dump,
)

pickle_load = _pickle_load
pickle_dump = _pickle_dump


def cal_missing_rate(X):
    return corruptor.cal_missing_rate(X)


def masked_fill(X, mask, val):
    return corruptor.masked_fill(X, mask, val)


def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)


def torch_parse_delta(missing_mask):
    """Generate time-gap (delta) matrix from missing masks. Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : torch.tensor, shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, torch.tensor,
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask):
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


def numpy_parse_delta(missing_mask):
    """Generate time-gap (delta) matrix from missing masks. Please refer to :cite:`che2018GRUD` for its math definition.

    Parameters
    ----------
    missing_mask : np.ndarray, shape of [n_steps, n_features] or [n_samples, n_steps, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    delta, np.ndarray,
        Delta matrix indicates time gaps of missing values.
    """

    def cal_delta_for_single_sample(mask):
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
