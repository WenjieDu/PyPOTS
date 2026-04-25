"""
Test cases for Linear Interpolation(Lerp) imputation method.
"""

# Created by Cole Sussmeier <colesussmeier@gmail.com>
# License: BSD-3-Clause


import unittest

import numpy as np
import pytest
import torch

from pypots.imputation import Lerp
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse
from tests.global_test_config import (
    DATA,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
)


class TestLerp(unittest.TestCase):
    logger.info("Running tests for an imputation model Lerp...")
    lerp = Lerp()

    @pytest.mark.xdist_group(name="imputation-lerp")
    def test_0_impute(self):
        # if input data is numpy ndarray
        test_X_imputed = self.lerp.predict(TEST_SET)["imputation"]
        assert not np.isnan(test_X_imputed).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(test_X_imputed, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        logger.info(f"Lerp test_MSE: {test_MSE}")

        # if input data is torch tensor
        X = torch.from_numpy(np.copy(TEST_SET["X"]))
        test_X_ori = torch.from_numpy(np.copy(DATA["test_X_ori"]))
        test_X_indicating_mask = torch.from_numpy(np.copy(DATA["test_X_indicating_mask"]))

        test_X_imputed = self.lerp.predict({"X": X})["imputation"]
        assert not torch.isnan(test_X_imputed).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(test_X_imputed, test_X_ori, test_X_indicating_mask)
        logger.info(f"Lerp test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-lerp")
    def test_1_list_input(self):
        """Test that list input with missing values is converted and imputed correctly."""
        X = np.random.randn(5, 10, 3)
        X[0, 2, 0] = np.nan
        X[1, 5, 1] = np.nan
        X[2, 8, 2] = np.nan
        X_list = X.tolist()
        result = self.lerp.predict({"X": X_list})["imputation"]
        assert not np.isnan(result).any(), "List input with NaN should be converted and interpolated."

    @pytest.mark.xdist_group(name="imputation-lerp")
    def test_4_lazy_loading(self):
        self.lerp.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.lerp.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading Lerp test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
