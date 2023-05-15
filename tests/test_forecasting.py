"""
Test cases for forecasting models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import pytest

from pypots.data.generating import gene_incomplete_random_walk_dataset
from pypots.forecasting import BTTF
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae

EPOCHS = 5
DATA = gene_incomplete_random_walk_dataset(n_steps=120, n_features=37)
TEST_SET = {"X": DATA["test_X"][:, :100]}


class TestBTTF(unittest.TestCase):
    logger.info("Running tests for a forecasting model BTTF...")

    # initialize a BTTF model
    bttf = BTTF(
        n_steps=100,
        n_features=10,
        pred_step=20,
        rank=10,
        time_lags=[1, 2, 3, 10, 10 + 1, 10 + 2, 20, 20 + 1, 20 + 2],
        burn_iter=5,
        gibbs_iter=5,
        multi_step=1,
    )

    @pytest.mark.xdist_group(name="forecasting-bttf")
    def test_0_forecasting(self):
        predictions = self.bttf.forecast(TEST_SET)
        logger.info(f"prediction shape: {predictions.shape}")
        mae = cal_mae(predictions, DATA["test_X_intact"][:, 100:])
        logger.info(f"prediction MAE: {mae}")


if __name__ == "__main__":
    unittest.main()
