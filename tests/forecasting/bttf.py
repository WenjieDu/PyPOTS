"""
Test cases for BTTF forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import pytest

from pypots.forecasting import BTTF
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse
from tests.global_test_config import DATA, FORECASTING_TEST_SET, N_PRED_STEPS


class TestBTTF(unittest.TestCase):
    logger.info("Running tests for a forecasting model BTTF...")

    # initialize a BTTF model
    bttf = BTTF(
        n_steps=DATA["n_steps"] - N_PRED_STEPS,
        n_features=DATA["n_features"],
        pred_step=N_PRED_STEPS,
        rank=2,
        time_lags=[1, 2, 3, 2, 2 + 1, 2 + 2, 3, 3 + 1, 3 + 1],
        burn_iter=2,
        gibbs_iter=2,
        gamma=5,
        multi_step=1,
    )

    @pytest.mark.xdist_group(name="forecasting-bttf")
    def test_0_forecasting(self):
        predictions = self.bttf.predict(FORECASTING_TEST_SET)["forecasting"]
        mse = calc_mse(predictions, FORECASTING_TEST_SET["X_pred"])
        logger.info(f"prediction MSE: {mse}")


if __name__ == "__main__":
    unittest.main()
