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
from pypots.utils.metrics import cal_mae
from tests.global_test_config import (
    DATA,
)
from tests.imputation.config import (
    TEST_SET,
)


class TestLOCF(unittest.TestCase):
    logger.info("Running tests for an imputation model LOCF...")
    locf_zero = LOCF(first_step_imputation="zero")
    locf_backward = LOCF(first_step_imputation="backward")
    locf_mean = LOCF(first_step_imputation="mean")
    locf_nan = LOCF(first_step_imputation="nan")

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_0_impute(self):
        TEST_SET["X"][0, 0, 0] = np.nan
        # if input data is numpy ndarray
        test_X_imputed_zero = self.locf_zero.predict(TEST_SET)["imputation"]
        assert not np.isnan(
            test_X_imputed_zero
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed_zero, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"LOCF (zero) test_MAE: {test_MAE}")

        test_X_imputed_backward = self.locf_backward.predict(TEST_SET)["imputation"]
        assert not np.isnan(
            test_X_imputed_backward
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed_backward,
            DATA["test_X_intact"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"LOCF (backward) test_MAE: {test_MAE}")

        test_X_imputed_mean = self.locf_mean.predict(TEST_SET)["imputation"]
        assert not np.isnan(
            test_X_imputed_mean
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed_mean,
            DATA["test_X_intact"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"LOCF (mean) test_MAE: {test_MAE}")

        test_X_imputed_nan = self.locf_nan.predict(TEST_SET)["imputation"]
        num_of_missing = np.isnan(test_X_imputed_nan).sum()
        assert num_of_missing > 0, "Output should have missing data but not."
        logger.info(f"LOCF (nan) still have {num_of_missing} missing values.")

        # if input data is torch tensor
        TEST_SET["X"] = torch.from_numpy(TEST_SET["X"])
        DATA["test_X_intact"] = torch.from_numpy(DATA["test_X_intact"])
        DATA["test_X_indicating_mask"] = torch.from_numpy(
            DATA["test_X_indicating_mask"]
        )
        test_X_imputed_zero = self.locf_zero.predict(TEST_SET)["imputation"]
        assert not torch.isnan(
            test_X_imputed_zero
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed_zero, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"LOCF (zero) test_MAE: {test_MAE}")

        test_X_imputed_backward = self.locf_backward.predict(TEST_SET)["imputation"]
        assert not torch.isnan(
            test_X_imputed_backward
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed_backward,
            DATA["test_X_intact"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"LOCF (backward) test_MAE: {test_MAE}")

        test_X_imputed_mean = self.locf_mean.predict(TEST_SET)["imputation"]
        assert not torch.isnan(
            test_X_imputed_mean
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed_mean,
            DATA["test_X_intact"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"LOCF (mean) test_MAE: {test_MAE}")

        test_X_imputed_nan = self.locf_nan.predict(TEST_SET)["imputation"]
        num_of_missing = torch.isnan(test_X_imputed_nan).sum()
        assert num_of_missing > 0, "Output should have missing data but not."
        logger.info(f"LOCF (nan) still have {num_of_missing} missing values.")


if __name__ == "__main__":
    unittest.main()
