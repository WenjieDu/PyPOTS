"""
Test cases for SAITS imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import SAITS
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mse
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    H5_TRAIN_SET_PATH,
    H5_VAL_SET_PATH,
    H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestSAITS(unittest.TestCase):
    logger.info("Running tests for an imputation model SAITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "SAITS")
    model_save_name = "saved_saits_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=32,
        d_ffn=32,
        n_heads=2,
        d_k=16,
        d_v=16,
        dropout=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_1_impute(self):
        imputation_results = self.saits.predict(TEST_SET, return_latent_vars=True)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."
        assert (
            "latent_vars" in imputation_results.keys()
        ), "Latent variables are not returned thought `return_latent_vars` is set as True."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"SAITS test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_2_parameters(self):
        assert hasattr(self.saits, "model") and self.saits.model is not None

        assert hasattr(self.saits, "optimizer") and self.saits.optimizer is not None

        assert hasattr(self.saits, "best_loss")
        self.assertNotEqual(self.saits.best_loss, float("inf"))

        assert (
            hasattr(self.saits, "best_model_dict")
            and self.saits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.saits)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.saits.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.saits.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_4_lazy_loading(self):
        self.saits.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        imputation_results = self.saits.predict(H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading SAITS test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
