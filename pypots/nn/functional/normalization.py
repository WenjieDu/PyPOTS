"""
Store normalization functions for neural networks.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Tuple, Optional

import torch


def nonstationary_norm(
    X: torch.Tensor,
    missing_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalization from Non-stationary Transformer. Please refer to :cite:`liu2022nonstationary` for more details.

    Parameters
    ----------
    X : torch.Tensor
        Input data to be normalized. Shape: (n_samples, n_steps (seq_len), n_features).

    missing_mask : torch.Tensor, optional
        Missing mask has the same shape as X. 1 indicates observed and 0 indicates missing.

    Returns
    -------
    X_enc : torch.Tensor
        Normalized data. Shape: (n_samples, n_steps (seq_len), n_features).

    means : torch.Tensor
        Means values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    stdev : torch.Tensor
        Standard deviation values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    """
    if torch.isnan(X).any():
        if missing_mask is None:
            missing_mask = torch.isnan(X)
        else:
            raise ValueError("missing_mask is given but X still contains nan values.")

    if missing_mask is None:
        means = X.mean(1, keepdim=True).detach()
        X_enc = X - means
        variance = torch.var(X_enc, dim=1, keepdim=True, unbiased=False) + 1e-9
        stdev = torch.sqrt(variance).detach()
    else:
        # for data contain missing values, add a small number to avoid dividing by 0
        missing_sum = torch.sum(missing_mask == 1, dim=1, keepdim=True) + 1e-9
        means = torch.sum(X, dim=1, keepdim=True) / missing_sum
        X_enc = X - means
        X_enc = X_enc.masked_fill(missing_mask == 0, 0)
        variance = torch.sum(X_enc * X_enc, dim=1, keepdim=True) + 1e-9
        stdev = torch.sqrt(variance / missing_sum)

    X_enc /= stdev
    return X_enc, means, stdev


def nonstationary_denorm(
    X: torch.Tensor,
    means: torch.Tensor,
    stdev: torch.Tensor,
) -> torch.Tensor:
    """De-Normalization from Non-stationary Transformer. Please refer to :cite:`liu2022nonstationary` for more details.

    Parameters
    ----------
    X : torch.Tensor
        Input data to be de-normalized. Shape: (n_samples, n_steps (seq_len), n_features).

    means : torch.Tensor
        Means values for de-normalization . Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    stdev : torch.Tensor
        Standard deviation values for de-normalization. Shape: (n_samples, n_features) or (n_samples, 1, n_features).

    Returns
    -------
    X_denorm : torch.Tensor
        De-normalized data. Shape: (n_samples, n_steps (seq_len), n_features).

    """
    assert (
        len(X) == len(means) == len(stdev)
    ), "Input data and normalization parameters should have the same number of samples."
    if len(means.shape) == 2:
        means = means.unsqueeze(1)
    if len(stdev.shape) == 2:
        stdev = stdev.unsqueeze(1)

    X = X * stdev  # (stdev.repeat(1, n_steps, 1))
    X = X + means  # (means.repeat(1, n_steps, 1))
    return X
