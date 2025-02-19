"""
Test cases for TRMF imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import TRMF
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse
from pypots.utils.visual.data import plot_data, plot_missingness
from tests.global_test_config import (
    DATA,
    TRAIN_SET,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestTRMF(unittest.TestCase):
    logger.info("Running tests for an imputation model TRMF...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "TRMF")
    model_save_name = "saved_trmf_model.pypots"

    # initialize a TRMF model
    trmf = TRMF(
        (1, 5),
        K=8,
        lambda_f=1,
        lambda_x=1,
        lambda_w=1,
        alpha=1,
        eta=1000,
        max_iter=1000,
        saving_path=saving_path,
    )

    @pytest.mark.xdist_group(name="imputation-trmf")
    def test_0_fit(self):
        self.trmf.fit(TRAIN_SET)

    @pytest.mark.xdist_group(name="imputation-trmf")
    def test_1_impute(self):
        imputation_results = self.trmf.predict(TRAIN_SET, return_latent_vars=True)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["train_X_ori"],
            np.isnan(DATA["train_X"]) ^ np.isnan(DATA["train_X_ori"]),
        )
        logger.info(f"TRMF test_MSE: {test_MSE}")

        # plot the missingness and imputed data
        plot_missingness(~np.isnan(TEST_SET["X"]), 0, imputation_results["imputation"].shape[1])
        plot_data(TEST_SET["X"], TEST_SET["X_ori"], imputation_results["imputation"])

    @pytest.mark.xdist_group(name="imputation-trmf")
    def test_2_parameters(self):
        assert hasattr(self.trmf, "model") and self.trmf.model is not None

    @pytest.mark.xdist_group(name="imputation-trmf")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.trmf)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.trmf.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.trmf.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-trmf")
    def test_4_lazy_loading(self):
        self.trmf.fit(GENERAL_H5_TRAIN_SET_PATH)
        imputation_results = self.trmf.predict(GENERAL_H5_TRAIN_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["train_X_ori"],
            np.isnan(DATA["train_X"]) ^ np.isnan(DATA["train_X_ori"]),
        )
        logger.info(f"Lazy-loading TRMF test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
