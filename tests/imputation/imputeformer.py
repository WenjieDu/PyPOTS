"""
Test cases for ImputeFormer imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import ImputeFormer
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
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestImputeFormer(unittest.TestCase):
    logger.info("Running tests for an imputation model ImputeFormer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "ImputeFormer")
    model_save_name = "saved_imputeformer_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a ImputeFormer model
    imputeformer = ImputeFormer(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_input_embed=32,
        d_learnable_embed=32,
        d_proj=32,
        d_ffn=32,
        n_temporal_heads=2,
        dropout=0,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-imputeformer")
    def test_0_fit(self):
        self.imputeformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-imputeformer")
    def test_1_impute(self):
        imputation_results = self.imputeformer.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"ImputeFormer test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-imputeformer")
    def test_2_parameters(self):
        assert (
            hasattr(self.imputeformer, "model") and self.imputeformer.model is not None
        )

        assert (
            hasattr(self.imputeformer, "optimizer")
            and self.imputeformer.optimizer is not None
        )

        assert hasattr(self.imputeformer, "best_loss")
        self.assertNotEqual(self.imputeformer.best_loss, float("inf"))

        assert (
            hasattr(self.imputeformer, "best_model_dict")
            and self.imputeformer.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-imputeformer")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.imputeformer)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.imputeformer.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.imputeformer.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-imputeformer")
    def test_4_lazy_loading(self):
        self.imputeformer.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.imputeformer.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading ImputeFormer test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
