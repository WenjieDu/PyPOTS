"""
Test cases for TimeLLM imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import TimeLLM
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


@pytest.mark.xdist_group(name="imputation-timellm")
class TestTimeLLM(unittest.TestCase):
    logger.info("Running tests for an imputation model TimeLLM...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "TimeLLM")
    model_save_name = "saved_timellm_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a TimeLLM model
    timellm = TimeLLM(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=32,
        patch_len=2,
        stride=2,
        d_model=32,
        d_ffn=128,
        d_llm=768,
        n_heads=8,
        llm_model_type="GPT2",
        dropout=0.1,
        domain_prompt_content="The next value in a random walk is a modification of the previous value in the "
        f'sequence. This dataset contains {DATA["n_steps"]} steps of a random walk with {DATA["n_features"]} features.',
        batch_size=8,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-timellm")
    def test_0_fit(self):
        self.timellm.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-timellm")
    def test_1_impute(self):
        imputation_results = self.timellm.predict(TEST_SET)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"TimeLLM test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-timellm")
    def test_2_parameters(self):
        assert hasattr(self.timellm, "model") and self.timellm.model is not None

        assert hasattr(self.timellm, "optimizer") and self.timellm.optimizer is not None

        assert hasattr(self.timellm, "best_loss")
        self.assertNotEqual(self.timellm.best_loss, float("inf"))

        assert hasattr(self.timellm, "best_model_dict") and self.timellm.best_model_dict is not None

    @pytest.mark.xdist_group(name="imputation-timellm")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.timellm)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.timellm.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.timellm.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-timellm")
    def test_4_lazy_loading(self):
        self.timellm.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.timellm.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(
            imputation_results["imputation"]
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputation_results["imputation"],
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading TimeLLM test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
