"""
Test cases for normalization functions.
"""

# License: BSD-3-Clause

import pytest
import torch

from pypots.nn.functional import nonstationary_denorm, nonstationary_norm


def test_nonstationary_norm_infers_observed_mask_from_nans() -> None:
    X_raw = torch.tensor(
        [
            [[1.0, 10.0], [float("nan"), 20.0], [3.0, float("nan")], [5.0, 40.0]],
            [[2.0, float("nan")], [4.0, 6.0], [6.0, 8.0], [float("nan"), 10.0]],
        ],
        dtype=torch.float64,
    )
    observed_mask = (~torch.isnan(X_raw)).to(X_raw.dtype)
    X_zeroed = torch.nan_to_num(X_raw, nan=0.0)

    X_enc_nan, means_nan, stdev_nan = nonstationary_norm(X_raw)
    X_enc_mask, means_mask, stdev_mask = nonstationary_norm(X_zeroed, observed_mask)

    assert not torch.isnan(X_enc_nan).any()
    assert not torch.isnan(means_nan).any()
    assert not torch.isnan(stdev_nan).any()
    torch.testing.assert_close(X_enc_nan, X_enc_mask)
    torch.testing.assert_close(means_nan, means_mask)
    torch.testing.assert_close(stdev_nan, stdev_mask)

    # Observed entries round-trip through denormalization, and missing entries stay zero in X_enc.
    expected_means = torch.tensor([[[3.0, 70.0 / 3.0]], [[4.0, 8.0]]], dtype=torch.float64)
    torch.testing.assert_close(means_nan, expected_means, atol=1e-6, rtol=1e-6)
    assert torch.all(X_enc_nan[observed_mask == 0] == 0)
    reconstructed = nonstationary_denorm(X_enc_nan, means_nan, stdev_nan)
    torch.testing.assert_close(reconstructed[observed_mask == 1], X_raw[observed_mask == 1])


def test_nonstationary_norm_ignores_nonzero_masked_entries() -> None:
    X_zeroed = torch.tensor([[[2.0, 0.0], [4.0, 10.0], [0.0, 20.0], [6.0, 30.0]]], dtype=torch.float64)
    X_filled = torch.tensor([[[2.0, 999.0], [4.0, 10.0], [-500.0, 20.0], [6.0, 30.0]]], dtype=torch.float64)
    missing_mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float64)

    X_enc_filled, means_filled, stdev_filled = nonstationary_norm(X_filled, missing_mask)
    X_enc_zeroed, means_zeroed, stdev_zeroed = nonstationary_norm(X_zeroed, missing_mask)

    torch.testing.assert_close(X_enc_filled, X_enc_zeroed)
    torch.testing.assert_close(means_filled, means_zeroed)
    torch.testing.assert_close(stdev_filled, stdev_zeroed)
    expected_means = torch.tensor([[[4.0, 20.0]]], dtype=torch.float64)
    torch.testing.assert_close(means_filled, expected_means, atol=1e-6, rtol=1e-6)


def test_nonstationary_norm_all_nan_feature_is_finite() -> None:
    X = torch.tensor([[[float("nan"), 2.0], [float("nan"), 4.0]]], dtype=torch.float64)

    X_enc, means, stdev = nonstationary_norm(X)

    assert torch.isfinite(X_enc).all()
    assert torch.isfinite(means).all()
    assert torch.isfinite(stdev).all()
    assert torch.all(X_enc[:, :, 0] == 0)
    torch.testing.assert_close(means[:, :, 1], torch.tensor([[3.0]], dtype=torch.float64), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(X_enc[:, :, 1], torch.tensor([[-1.0, 1.0]], dtype=torch.float64), atol=1e-6, rtol=1e-6)


def test_nonstationary_norm_raises_when_nan_and_mask_both_given() -> None:
    X = torch.tensor([[[1.0], [float("nan")]]], dtype=torch.float32)
    missing_mask = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)

    with pytest.raises(ValueError, match="missing_mask is given but X still contains nan values"):
        nonstationary_norm(X, missing_mask)
