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
from pypots.utils.metrics import calc_mae
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
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
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
        test_MAE = calc_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"Transformer test_MAE: {test_MAE}")

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


if __name__ == "__main__":
    unittest.main()
