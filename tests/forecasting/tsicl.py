"""
Test cases for TSICL forecasting model.
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>,
# and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

import os.path
import unittest

import numpy as np
import pytest

from pypots.forecasting import TSICL
from pypots.nn.functional import calc_mse
from pypots.utils.logging import logger
from tests.global_test_config import (
    DEVICE,
    FORECASTING_H5_TEST_SET_PATH,
    FORECASTING_TEST_SET,
    N_PRED_STEPS,
    RESULT_SAVING_DIR_FOR_FORECASTING,
    check_tb_and_model_checkpoints_existence,
)


class TestTSICL(unittest.TestCase):
    logger.info("Running tests for a forecasting model TSICL...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_FORECASTING, "TSICL")
    model_save_name = "saved_tsicl_model.pypots"

    tsicl = TSICL(n_pred_steps=N_PRED_STEPS, device=DEVICE, saving_path=saving_path)

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_0_fit(self):
        with pytest.warns(UserWarning):
            self.tsicl.fit(FORECASTING_TEST_SET)

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_1_forecasting(self):
        forecasting_X = self.tsicl.predict(FORECASTING_TEST_SET)["forecasting"]
        assert forecasting_X.shape == FORECASTING_TEST_SET["X_pred"].shape
        assert not np.isnan(forecasting_X).any(), (
            "Output has missing values in the forecasting results that should not be."
        )
        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"TSICL test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_2_long_horizon_rollout(self):
        # forecasting beyond the model's per-pass horizon cap rolls the context forward autoregressively
        n_pred_steps = self.tsicl.max_target_length + 8
        tsicl = TSICL(n_pred_steps=n_pred_steps, device=DEVICE)
        X = FORECASTING_TEST_SET["X"][:1]
        forecasting_X = tsicl.predict({"X": X})["forecasting"]
        assert forecasting_X.shape == (1, n_pred_steps, X.shape[-1])
        assert not np.isnan(forecasting_X).any()

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_3_fully_missing_series(self):
        # a fully-missing series (even alone in its batch) gives the model no context, it must fall back
        # to 0 instead of crashing
        X = FORECASTING_TEST_SET["X"][:2].copy()
        X[0, :, 0] = np.nan
        tsicl = TSICL(n_pred_steps=N_PRED_STEPS, device=DEVICE, batch_size=1)
        forecasting_X = tsicl.predict({"X": X})["forecasting"]
        assert not np.isnan(forecasting_X).any()
        assert (forecasting_X[0, :, 0] == 0).all()

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_4_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.tsicl)

        # save the model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.tsicl.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.tsicl.load(saved_model_path)

    @pytest.mark.xdist_group(name="forecasting-tsicl")
    def test_5_lazy_loading(self):
        forecasting_results = self.tsicl.predict(FORECASTING_H5_TEST_SET_PATH)
        forecasting_X = forecasting_results["forecasting"]
        assert not np.isnan(forecasting_X).any(), (
            "Output has missing values in the forecasting results that should not be."
        )

        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"Lazy-loading TSICL test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
