"""
Test cases for error metric functions with NaN target support.
"""

# Created by Haoyu Wang
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest
import torch

from pypots.nn.functional.error import calc_mae, calc_mse, calc_rmse, calc_mre


class TestErrorMetricsNaNTargets(unittest.TestCase):
    """Tests for NaN-tolerant metric computation when masks are provided."""

    def test_backward_compat_nan_free(self):
        """Verify NaN-free inputs produce the same results as before."""
        p = np.array([1.0, 2.0, 1.0, 4.0, 6.0])
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mae = calc_mae(p, t)
        assert abs(mae - 0.6) < 1e-6, f"Expected 0.6, got {mae}"

    def test_backward_compat_with_masks(self):
        """Verify NaN-free inputs with masks produce the same results."""
        p = np.array([1.0, 2.0, 1.0, 4.0, 6.0])
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        mae = calc_mae(p, t, m)
        assert abs(mae - 0.5) < 1e-6, f"Expected 0.5, got {mae}"

    def test_nan_targets_with_masks_numpy(self):
        """NaN positions in targets should be auto-excluded via mask."""
        p = np.array([1.0, 2.0, 1.0, 4.0, 6.0])
        t = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        m = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # Valid positions: 0, 2, 3 → errors: |1-1|+|1-3|+|4-4| = 2, count=3
        mae = calc_mae(p, t, m)
        expected = 2.0 / 3.0
        assert abs(mae - expected) < 1e-6, f"Expected {expected}, got {mae}"

    def test_nan_targets_with_masks_torch(self):
        """Same as above but with PyTorch tensors."""
        p = torch.tensor([1.0, 2.0, 1.0, 4.0, 6.0])
        t = torch.tensor([1.0, float("nan"), 3.0, 4.0, float("nan")])
        m = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        mae = calc_mae(p, t, m)
        expected = 2.0 / 3.0
        assert abs(mae.item() - expected) < 1e-6

    def test_nan_targets_without_masks_raises(self):
        """NaN targets without masks should raise ValueError."""
        p = np.array([1.0, 2.0, 3.0])
        t = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="no `masks` were provided"):
            calc_mae(p, t)

    def test_all_nan_targets(self):
        """When all targets are NaN, all positions are masked → metric ≈ 0."""
        p = np.array([1.0, 2.0, 3.0])
        t = np.array([np.nan, np.nan, np.nan])
        m = np.array([1.0, 1.0, 1.0])
        mae = calc_mae(p, t, m)
        assert abs(mae) < 1e-6, f"Expected ~0, got {mae}"

    def test_no_nan_propagation_mse_rmse_mre(self):
        """Verify MSE, RMSE, MRE also handle NaN targets without propagation."""
        p = np.array([1.0, 2.0, 1.0, 4.0, 6.0])
        t = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        m = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mse = calc_mse(p, t, m)
        rmse = calc_rmse(p, t, m)
        mre = calc_mre(p, t, m)
        assert not np.isnan(mse), "MSE should not be NaN"
        assert not np.isnan(rmse), "RMSE should not be NaN"
        assert not np.isnan(mre), "MRE should not be NaN"

    def test_predictions_nan_still_rejected(self):
        """Predictions with NaN should still raise AssertionError."""
        p = np.array([1.0, np.nan, 3.0])
        t = np.array([1.0, 2.0, 3.0])
        m = np.array([1.0, 1.0, 1.0])
        with pytest.raises(AssertionError, match="predictions"):
            calc_mae(p, t, m)

    def test_masks_already_exclude_nan(self):
        """When masks already exclude NaN positions, result is unchanged."""
        p = np.array([1.0, 2.0, 1.0, 4.0, 6.0])
        t = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        m_auto = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        m_manual = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        mae_auto = calc_mae(p, t, m_auto)
        mae_manual = calc_mae(p, t, m_manual)
        assert abs(mae_auto - mae_manual) < 1e-6


if __name__ == "__main__":
    unittest.main()
