"""
Test cases for FiLM forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.forecasting import FiLM
from pypots.nn.functional import calc_mse
from pypots.optim import Adam
from pypots.utils.logging import logger
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


class TestFiLM(unittest.TestCase):
    logger.info("Running tests for a forecasting model FiLM...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_FORECASTING, "FiLM")
    model_save_name = "saved_film_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a FiLM model
    film = FiLM(
        n_steps=DATA["n_steps"] - N_PRED_STEPS,
        n_features=DATA["n_features"],
        n_pred_steps=N_PRED_STEPS,
        n_pred_features=DATA["n_features"],
        window_size=[2],
        multiscale=[1, 2],
        modes1=512,
        dropout=0.5,
        d_model=512,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="forecasting-film")
    def test_0_fit(self):
        self.film.fit(FORECASTING_TRAIN_SET, FORECASTING_VAL_SET)

    @pytest.mark.xdist_group(name="forecasting-film")
    def test_1_forecasting(self):
        forecasting_X = self.film.predict(FORECASTING_TEST_SET)["forecasting"]
        assert not np.isnan(
            forecasting_X
        ).any(), "Output has missing values in the forecasting results that should not be."
        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"FiLM test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="forecasting-film")
    def test_2_parameters(self):
        assert hasattr(self.film, "model") and self.film.model is not None

        assert hasattr(self.film, "optimizer") and self.film.optimizer is not None

        assert hasattr(self.film, "best_loss")
        self.assertNotEqual(self.film.best_loss, float("inf"))

        assert hasattr(self.film, "best_model_dict") and self.film.best_model_dict is not None

    @pytest.mark.xdist_group(name="forecasting-film")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.film)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.film.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.film.load(saved_model_path)

    @pytest.mark.xdist_group(name="forecasting-film")
    def test_4_lazy_loading(self):
        self.film.fit(FORECASTING_H5_TRAIN_SET_PATH, FORECASTING_H5_VAL_SET_PATH)
        forecasting_results = self.film.predict(FORECASTING_H5_TEST_SET_PATH)
        forecasting_X = forecasting_results["forecasting"]
        assert not np.isnan(
            forecasting_X
        ).any(), "Output has missing values in the forecasting results that should not be."

        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"Lazy-loading FiLM test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
