"""
Test cases for LOCF imputation method.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest

import numpy as np
import pytest

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
    locf = LOCF(nan=0)

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_0_impute(self):
        test_X_imputed = self.locf.impute(TEST_SET)
        assert not np.isnan(
            test_X_imputed
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"LOCF test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_1_parameters(self):
        assert hasattr(self.locf, "nan") and self.locf.nan is not None


if __name__ == "__main__":
    unittest.main()
