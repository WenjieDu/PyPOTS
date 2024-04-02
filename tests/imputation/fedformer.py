"""
Test cases for FEDformer imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import FEDformer
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


class TestFEDformer(unittest.TestCase):
    logger.info("Running tests for an imputation model FEDformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "FEDformer")
    model_save_name = "saved_fedformer_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a FEDformer model
    fedformer = FEDformer(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ffn=32,
        moving_avg_window_size=3,
        dropout=0,
        version="Wavelets",  # TODO: Fourier version does not work
        modes=32,
        mode_select="random",
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-fedformer")
    def test_0_fit(self):
        self.fedformer.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-fedformer")
    def test_1_impute(self):
        imputation_results = self.fedformer.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"FEDformer test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-fedformer")
    def test_2_parameters(self):
        assert hasattr(self.fedformer, "model") and self.fedformer.model is not None

        assert (
            hasattr(self.fedformer, "optimizer")
            and self.fedformer.optimizer is not None
        )

        assert hasattr(self.fedformer, "best_loss")
        self.assertNotEqual(self.fedformer.best_loss, float("inf"))

        assert (
            hasattr(self.fedformer, "best_model_dict")
            and self.fedformer.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-fedformer")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.fedformer)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.fedformer.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.fedformer.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-fedformer")
    def test_4_lazy_loading(self):
        self.fedformer.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        imputation_results = self.fedformer.predict(H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading FEDformer test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
