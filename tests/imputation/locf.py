"""
Test cases for LOCF imputation method.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest
import torch

from pypots.imputation import LOCF
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse
from tests.global_test_config import (
    DATA,
    DEVICE,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
)


class TestLOCF(unittest.TestCase):
    logger.info("Running tests for an imputation model LOCF...")
    locf_zero = LOCF(first_step_imputation="zero", device=DEVICE)
    locf_backward = LOCF(first_step_imputation="backward", device=DEVICE)
    locf_median = LOCF(first_step_imputation="median", device=DEVICE)
    locf_nan = LOCF(first_step_imputation="nan", device=DEVICE)

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_0_impute(self):
        # if input data is numpy ndarray
        test_X_imputed_zero = self.locf_zero.predict(TEST_SET)["imputation"]
        assert not np.isnan(test_X_imputed_zero).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(test_X_imputed_zero, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        logger.info(f"LOCF (zero) test_MSE: {test_MSE}")

        test_X_imputed_backward = self.locf_backward.predict(TEST_SET)["imputation"]
        assert not np.isnan(test_X_imputed_backward).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(
            test_X_imputed_backward,
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"LOCF (backward) test_MSE: {test_MSE}")

        test_X_imputed_median = self.locf_median.predict(TEST_SET)["imputation"]
        assert not np.isnan(test_X_imputed_median).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(
            test_X_imputed_median,
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"LOCF (median) test_MSE: {test_MSE}")

        test_X_imputed_nan = self.locf_nan.predict(TEST_SET)["imputation"]
        num_of_missing = np.isnan(test_X_imputed_nan).sum()
        assert num_of_missing > 0, "Output should have missing data but not."
        logger.info(f"LOCF (nan) still have {num_of_missing} missing values.")

        # if input data is torch tensor
        X = torch.from_numpy(np.copy(TEST_SET["X"]))
        test_X_ori = torch.from_numpy(np.copy(DATA["test_X_ori"]))
        test_X_indicating_mask = torch.from_numpy(np.copy(DATA["test_X_indicating_mask"]))

        test_X_imputed_zero = self.locf_zero.predict({"X": X})["imputation"]
        assert not torch.isnan(test_X_imputed_zero).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(test_X_imputed_zero, test_X_ori, test_X_indicating_mask)
        logger.info(f"LOCF (zero) test_MSE: {test_MSE}")

        test_X_imputed_backward = self.locf_backward.predict({"X": X})["imputation"]
        assert not torch.isnan(test_X_imputed_backward).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(
            test_X_imputed_backward,
            test_X_ori,
            test_X_indicating_mask,
        )
        logger.info(f"LOCF (backward) test_MSE: {test_MSE}")

        test_X_imputed_median = self.locf_median.predict({"X": X})["imputation"]
        assert not torch.isnan(test_X_imputed_median).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(
            test_X_imputed_median,
            test_X_ori,
            test_X_indicating_mask,
        )
        logger.info(f"LOCF (median) test_MSE: {test_MSE}")

        test_X_imputed_nan = self.locf_nan.predict({"X": X})["imputation"]
        num_of_missing = torch.isnan(test_X_imputed_nan).sum()
        assert num_of_missing > 0, "Output should have missing data but not."
        logger.info(f"LOCF (nan) still have {num_of_missing} missing values.")

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_4_lazy_loading(self):
        self.locf_backward.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.locf_backward.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(imputation_results["imputation"]).any(), (
            "Output still has missing values after running impute()."
        )

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading LOCF test_MSE: {test_MSE}")


@pytest.mark.xdist_group(name="imputation-locf")
@pytest.mark.parametrize("strategy", ["zero", "backward", "median", "nan"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_torch_device(strategy, dtype, device):
    """Tensor imputation retains its device and handles leading or entire gaps."""
    X = torch.tensor(
        [
            [
                [np.nan, np.nan, np.nan],
                [2, np.nan, np.nan],
                [np.nan, 6, np.nan],
                [4, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        ],
        dtype=dtype,
        device=device,
    )
    original = X.clone()
    expected = torch.tensor(
        [[[np.nan, np.nan, np.nan], [2, np.nan, np.nan], [2, 6, np.nan], [4, 6, np.nan], [4, 6, np.nan]]],
        dtype=dtype,
        device=device,
    )
    if strategy == "zero":
        expected = torch.nan_to_num(expected)
    elif strategy in ["backward", "median"]:
        # Both first observations and the existing lower-middle medians are 2, 6.
        expected[:, :, 0] = torch.nan_to_num(expected[:, :, 0], nan=2)
        expected[:, :, 1] = torch.nan_to_num(expected[:, :, 1], nan=6)
        expected[:, :, 2] = 0
    actual = LOCF(first_step_imputation=strategy).predict({"X": X})["imputation"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(X, original, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
