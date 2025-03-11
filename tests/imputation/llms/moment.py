"""
Test cases for MOMENT imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import MOMENT
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


class TestMOMENT(unittest.TestCase):
    logger.info("Running tests for an imputation model MOMENT...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "MOMENT")
    model_save_name = "saved_moment_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a MOMENT model
    moment = MOMENT(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        patch_size=2,
        patch_stride=2,
        n_layers=1,
        d_ffn=512,
        dropout=0.1,
        d_model=512,
        transformer_backbone="t5-small",
        transformer_type="encoder_only",
        head_dropout=0.1,
        finetuning_mode="zero-shot",
        revin_affine=False,
        add_positional_embedding=False,
        value_embedding_bias=False,
        orth_gain=1.41,
        mask_ratio=0,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-moment")
    def test_0_fit(self):
        self.moment.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-moment")
    def test_1_impute(self):
        imputation_results = self.moment.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"MOMENT test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-moment")
    def test_2_parameters(self):
        assert hasattr(self.moment, "model") and self.moment.model is not None

        assert hasattr(self.moment, "optimizer") and self.moment.optimizer is not None

        assert hasattr(self.moment, "best_loss")
        self.assertNotEqual(self.moment.best_loss, float("inf"))

        assert hasattr(self.moment, "best_model_dict") and self.moment.best_model_dict is not None

    @pytest.mark.xdist_group(name="imputation-moment")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.moment)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.moment.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.moment.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-moment")
    def test_4_lazy_loading(self):
        self.moment.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.moment.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading MOMENT test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
