"""
Test cases for CSDI forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.forecasting import CSDI
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse, calc_quantile_crps

from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    N_PRED_STEPS,
    FORECASTING_TRAIN_SET,
    FORECASTING_VAL_SET,
    FORECASTING_TEST_SET,
    FORECASTING_H5_TRAIN_SET_PATH,
    FORECASTING_H5_VAL_SET_PATH,
    FORECASTING_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_FORECASTING,
    check_tb_and_model_checkpoints_existence,
)


class TestCSDI(unittest.TestCase):
    logger.info("Running tests for an forecasting model CSDI...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_FORECASTING, "CSDI")
    model_save_name = "saved_csdi_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a CSDI model
    csdi = CSDI(
        n_steps=DATA["n_steps"] - N_PRED_STEPS,
        n_features=DATA["n_features"],
        n_pred_steps=N_PRED_STEPS,
        n_pred_features=DATA["n_features"],
        n_layers=1,
        n_channels=8,
        d_time_embedding=32,
        d_feature_embedding=3,
        d_diffusion_embedding=32,
        n_diffusion_steps=5,
        n_heads=1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="forecasting-csdi")
    def test_0_fit(self):
        self.csdi.fit(FORECASTING_TRAIN_SET, FORECASTING_VAL_SET)

    @pytest.mark.xdist_group(name="forecasting-csdi")
    def test_1_forecasting(self):
        forecasting_X = self.csdi.predict(FORECASTING_TEST_SET, n_sampling_times=2)["forecasting"]
        test_CRPS = calc_quantile_crps(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        forecasting_X = forecasting_X.mean(axis=1)  # mean over sampling times
        assert not np.isnan(
            forecasting_X
        ).any(), "Output has missing values in the forecasting results that should not be."
        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"CSDI test_MSE: {test_MSE}, test_CRPS: {test_CRPS}")

    @pytest.mark.xdist_group(name="forecasting-csdi")
    def test_2_parameters(self):
        assert hasattr(self.csdi, "model") and self.csdi.model is not None

        assert hasattr(self.csdi, "optimizer") and self.csdi.optimizer is not None

        assert hasattr(self.csdi, "best_loss")
        self.assertNotEqual(self.csdi.best_loss, float("inf"))

        assert hasattr(self.csdi, "best_model_dict") and self.csdi.best_model_dict is not None

    @pytest.mark.xdist_group(name="forecasting-csdi")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.csdi)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.csdi.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.csdi.load(saved_model_path)

    @pytest.mark.xdist_group(name="forecasting-csdi")
    def test_4_lazy_loading(self):
        self.csdi.fit(FORECASTING_H5_TRAIN_SET_PATH, FORECASTING_H5_VAL_SET_PATH)
        forecasting_results = self.csdi.predict(FORECASTING_H5_TEST_SET_PATH)
        forecasting_X = forecasting_results["forecasting"]
        test_CRPS = calc_quantile_crps(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        forecasting_X = forecasting_X.mean(axis=1)  # mean over sampling times
        assert not np.isnan(
            forecasting_X
        ).any(), "Output has missing values in the forecasting results that should not be."

        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"Lazy-loading CSDI test_MSE: {test_MSE}, test_CRPS: {test_CRPS}")


if __name__ == "__main__":
    unittest.main()
