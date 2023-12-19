"""
Test cases for BTTF forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import pytest

from pypots.forecasting import BTTF
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae
from tests.forecasting.config import (
    TEST_SET,
    N_PRED_STEP,
)
from tests.global_test_config import DATA


class TestBTTF(unittest.TestCase):
    logger.info("Running tests for a forecasting model BTTF...")

    # initialize a BTTF model
    bttf = BTTF(
        n_steps=DATA["n_steps"] - N_PRED_STEP,
        n_features=10,
        pred_step=N_PRED_STEP,
        rank=10,
        time_lags=[1, 2, 3, 5, 5 + 1, 5 + 2, 10, 10 + 1, 10 + 2],
        burn_iter=5,
        gibbs_iter=5,
        multi_step=1,
    )

    @pytest.mark.xdist_group(name="forecasting-bttf")
    def test_0_forecasting(self):
        predictions = self.bttf.predict(TEST_SET)["forecasting"]
        mae = calc_mae(predictions, TEST_SET["X_ori"][:, -N_PRED_STEP:])
        logger.info(f"prediction MAE: {mae}")


if __name__ == "__main__":
    unittest.main()
