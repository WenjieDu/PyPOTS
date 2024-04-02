"""
Test cases for Transformer imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import Transformer
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


class TestTransformer(unittest.TestCase):
    logger.info("Running tests for an imputation model Transformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "Transformer")
    model_save_name = "saved_transformer_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a Transformer model
    transformer = Transformer(
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

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_0_fit(self):
        self.transformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_1_impute(self):
        imputed_X = self.transformer.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"Transformer test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_2_parameters(self):
        assert hasattr(self.transformer, "model") and self.transformer.model is not None

        assert (
            hasattr(self.transformer, "optimizer")
            and self.transformer.optimizer is not None
        )

        assert hasattr(self.transformer, "best_loss")
        self.assertNotEqual(self.transformer.best_loss, float("inf"))

        assert (
            hasattr(self.transformer, "best_model_dict")
            and self.transformer.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.transformer)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.transformer.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.transformer.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-transformer")
    def test_4_lazy_loading(self):
        self.transformer.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        imputation_results = self.transformer.predict(H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading Transformer test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
