"""
Test cases for TimeMixerPP imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import TimeMixerPP
from pypots.nn.functional import calc_mse
from pypots.optim import Adam
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestTimeMixerPP(unittest.TestCase):
    logger.info("Running tests for an imputation model TimeMixerPP...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "TimeMixerPP")
    model_save_name = "saved_timemixerpp_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a TimeMixerPP model
    timemixerpp = TimeMixerPP(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        top_k=5,
        d_model=16,
        d_ffn=16,
        n_heads=4,
        n_kernels=3,
        dropout=0.1,
        downsampling_window=1,
        downsampling_layers=2,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-timemixerpp")
    def test_0_fit(self):
        self.timemixerpp.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-timemixerpp")
    def test_1_impute(self):
        imputation_results = self.timemixerpp.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"TimeMixerPP test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-timemixerpp")
    def test_2_parameters(self):
        assert hasattr(self.timemixerpp, "model") and self.timemixerpp.model is not None

        assert hasattr(self.timemixerpp, "optimizer") and self.timemixerpp.optimizer is not None

        assert hasattr(self.timemixerpp, "best_loss")
        self.assertNotEqual(self.timemixerpp.best_loss, float("inf"))

        assert hasattr(self.timemixerpp, "best_model_dict") and self.timemixerpp.best_model_dict is not None

    @pytest.mark.xdist_group(name="imputation-timemixerpp")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.timemixerpp)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.timemixerpp.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.timemixerpp.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-timemixerpp")
    def test_4_lazy_loading(self):
        self.timemixerpp.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.timemixerpp.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading TimeMixerPP test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
