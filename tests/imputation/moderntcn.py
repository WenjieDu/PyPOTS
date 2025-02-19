"""
Test cases for ModernTCN imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import ModernTCN
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse
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


class TestModernTCN(unittest.TestCase):
    logger.info("Running tests for an imputation model ModernTCN...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "ModernTCN")
    model_save_name = "saved_moderntcn_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a ModernTCN model
    moderntcn = ModernTCN(
        DATA["n_steps"],
        DATA["n_features"],
        patch_size=3,
        patch_stride=2,
        downsampling_ratio=2,
        ffn_ratio=1,
        num_blocks=[1],
        large_size=[5],
        small_size=[3],
        dims=[32],
        small_kernel_merged=False,
        backbone_dropout=0.1,
        head_dropout=0.1,
        use_multi_scale=False,
        individual=False,
        apply_nonstationary_norm=False,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-moderntcn")
    def test_0_fit(self):
        self.moderntcn.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-moderntcn")
    def test_1_impute(self):
        imputation_results = self.moderntcn.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"ModernTCN test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-moderntcn")
    def test_2_parameters(self):
        assert hasattr(self.moderntcn, "model") and self.moderntcn.model is not None

        assert hasattr(self.moderntcn, "optimizer") and self.moderntcn.optimizer is not None

        assert hasattr(self.moderntcn, "best_loss")
        self.assertNotEqual(self.moderntcn.best_loss, float("inf"))

        assert hasattr(self.moderntcn, "best_model_dict") and self.moderntcn.best_model_dict is not None

    @pytest.mark.xdist_group(name="imputation-moderntcn")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.moderntcn)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.moderntcn.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.moderntcn.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-moderntcn")
    def test_4_lazy_loading(self):
        self.moderntcn.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.moderntcn.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading ModernTCN test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
