"""
Test cases for GPT4TS forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.forecasting import GPT4TS
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


class TestGPT4TS(unittest.TestCase):
    logger.info("Running tests for a forecasting model GPT4TS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_FORECASTING, "GPT4TS")
    model_save_name = "saved_gpt4ts_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a GPT4TS model
    gpt4ts = GPT4TS(
        n_steps=DATA["n_steps"] - N_PRED_STEPS,
        n_features=DATA["n_features"],
        n_pred_steps=N_PRED_STEPS,
        n_pred_features=DATA["n_features"],
        term="short",
        patch_size=DATA["n_steps"],
        patch_stride=8,
        n_layers=2,
        train_gpt_mlp=True,
        d_ffn=128,
        dropout=0.1,
        batch_size=8,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="forecasting-gpt4ts")
    def test_0_fit(self):
        self.gpt4ts.fit(FORECASTING_TRAIN_SET, FORECASTING_VAL_SET)

    @pytest.mark.xdist_group(name="forecasting-gpt4ts")
    def test_1_forecasting(self):
        forecasting_X = self.gpt4ts.predict(FORECASTING_TEST_SET)["forecasting"]
        assert not np.isnan(
            forecasting_X
        ).any(), "Output has missing values in the forecasting results that should not be."
        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"GPT4TS test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="forecasting-gpt4ts")
    def test_2_parameters(self):
        assert hasattr(self.gpt4ts, "model") and self.gpt4ts.model is not None

        assert hasattr(self.gpt4ts, "optimizer") and self.gpt4ts.optimizer is not None

        assert hasattr(self.gpt4ts, "best_loss")
        self.assertNotEqual(self.gpt4ts.best_loss, float("inf"))

        assert hasattr(self.gpt4ts, "best_model_dict") and self.gpt4ts.best_model_dict is not None

    @pytest.mark.xdist_group(name="forecasting-gpt4ts")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.gpt4ts)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.gpt4ts.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.gpt4ts.load(saved_model_path)

    @pytest.mark.xdist_group(name="forecasting-gpt4ts")
    def test_4_lazy_loading(self):
        self.gpt4ts.fit(FORECASTING_H5_TRAIN_SET_PATH, FORECASTING_H5_VAL_SET_PATH)
        forecasting_results = self.gpt4ts.predict(FORECASTING_H5_TEST_SET_PATH)
        forecasting_X = forecasting_results["forecasting"]
        assert not np.isnan(
            forecasting_X
        ).any(), "Output has missing values in the forecasting results that should not be."

        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"Lazy-loading GPT4TS test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
