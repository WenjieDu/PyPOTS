"""
Test cases for PatchTST imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import PatchTST
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


class TestPatchTST(unittest.TestCase):
    logger.info("Running tests for an imputation model PatchTST...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "PatchTST")
    model_save_name = "saved_patchtst_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a PatchTST model
    patchtst = PatchTST(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=64,
        d_ffn=32,
        n_heads=2,
        d_k=16,
        d_v=16,
        patch_len=16,
        stride=8,
        dropout=0.1,
        attn_dropout=0,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-patchtst")
    def test_0_fit(self):
        self.patchtst.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-patchtst")
    def test_1_impute(self):
        imputation_results = self.patchtst.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"PatchTST test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-patchtst")
    def test_2_parameters(self):
        assert hasattr(self.patchtst, "model") and self.patchtst.model is not None

        assert (
            hasattr(self.patchtst, "optimizer") and self.patchtst.optimizer is not None
        )

        assert hasattr(self.patchtst, "best_loss")
        self.assertNotEqual(self.patchtst.best_loss, float("inf"))

        assert (
            hasattr(self.patchtst, "best_model_dict")
            and self.patchtst.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-patchtst")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.patchtst)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.patchtst.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.patchtst.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-patchtst")
    def test_4_lazy_loading(self):
        self.patchtst.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        imputation_results = self.patchtst.predict(H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading PatchTST test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
