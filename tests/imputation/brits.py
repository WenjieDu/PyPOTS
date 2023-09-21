"""
Test cases for BRITS imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import BRITS
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae
from tests.global_test_config import (
    DATA,
    DEVICE,
    check_tb_and_model_checkpoints_existence,
)
from tests.imputation.config import (
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    EPOCHS,
)


class TestBRITS(unittest.TestCase):
    logger.info("Running tests for an imputation model BRITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "BRITS")
    model_save_name = "saved_BRITS_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a BRITS model
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_0_fit(self):
        self.brits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_1_impute(self):
        imputed_X = self.brits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"BRITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_2_parameters(self):
        assert hasattr(self.brits, "model") and self.brits.model is not None

        assert hasattr(self.brits, "optimizer") and self.brits.optimizer is not None

        assert hasattr(self.brits, "best_loss")
        self.assertNotEqual(self.brits.best_loss, float("inf"))

        assert (
            hasattr(self.brits, "best_model_dict")
            and self.brits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-brits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.brits)

        # save the trained model into file, and check if the path exists
        self.brits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.brits.load_model(saved_model_path)


if __name__ == "__main__":
    unittest.main()
