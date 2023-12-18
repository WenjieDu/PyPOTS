"""
Test cases for the optimizer SGD.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import numpy as np
import pytest

from pypots.imputation import SAITS
from pypots.optim import SGD
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae
from tests.global_test_config import DATA
from tests.optim.config import EPOCHS, TEST_SET, TRAIN_SET, VAL_SET


class TestSGD(unittest.TestCase):
    logger.info("Running tests for SGD...")

    # initialize a SGD optimizer
    sgd = SGD(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        d_model=128,
        d_inner=64,
        n_heads=2,
        d_k=64,
        d_v=64,
        dropout=0.1,
        optimizer=sgd,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="optim-sgd")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")


if __name__ == "__main__":
    unittest.main()
