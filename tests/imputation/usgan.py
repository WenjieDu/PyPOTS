"""
Test cases for US-GAN imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import USGAN
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


class TestUSGAN(unittest.TestCase):
    logger.info("Running tests for an imputation model US-GAN...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "US-GAN")
    model_save_name = "saved_USGAN_model.pypots"

    # initialize an Adam optimizer
    G_optimizer = Adam(lr=0.001, weight_decay=1e-5)
    D_optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a US-GAN model
    us_gan = USGAN(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCHS,
        saving_path=saving_path,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-usgan")
    def test_0_fit(self):
        self.us_gan.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-usgan")
    def test_1_impute(self):
        imputed_X = self.us_gan.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"US-GAN test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-usgan")
    def test_2_parameters(self):
        assert hasattr(self.us_gan, "model") and self.us_gan.model is not None

        assert (
            hasattr(self.us_gan, "G_optimizer") and self.us_gan.G_optimizer is not None
        )
        assert (
            hasattr(self.us_gan, "D_optimizer") and self.us_gan.D_optimizer is not None
        )

        assert hasattr(self.us_gan, "best_loss")
        self.assertNotEqual(self.us_gan.best_loss, float("inf"))

        assert (
            hasattr(self.us_gan, "best_model_dict")
            and self.us_gan.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-usgan")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.us_gan)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.us_gan.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.us_gan.load(saved_model_path)


if __name__ == "__main__":
    unittest.main()
