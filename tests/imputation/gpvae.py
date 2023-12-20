"""
Test cases for GP-VAE imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import GPVAE
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


class TestGPVAE(unittest.TestCase):
    logger.info("Running tests for an imputation model GP-VAE...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "GP-VAE")
    model_save_name = "saved_GPVAE_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a GP-VAE model
    gp_vae = GPVAE(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-gpvae")
    def test_0_fit(self):
        self.gp_vae.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-gpvae")
    def test_1_impute(self):
        imputed_X = self.gp_vae.predict(TEST_SET, n_sampling_times=2)["imputation"]
        imputed_X = imputed_X.mean(axis=1)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(
            imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"GP-VAE test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-gpvae")
    def test_2_parameters(self):
        assert hasattr(self.gp_vae, "model") and self.gp_vae.model is not None

        assert hasattr(self.gp_vae, "optimizer") and self.gp_vae.optimizer is not None

        assert hasattr(self.gp_vae, "best_loss")
        self.assertNotEqual(self.gp_vae.best_loss, float("inf"))

        assert (
            hasattr(self.gp_vae, "best_model_dict")
            and self.gp_vae.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-gpvae")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.gp_vae)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.gp_vae.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.gp_vae.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-gpvae")
    def test_4_lazy_loading(self):
        self.gp_vae.fit(H5_TRAIN_SET_PATH, H5_VAL_SET_PATH)
        imputed_X = self.gp_vae.predict(H5_TEST_SET_PATH, n_sampling_times=2)[
            "imputation"
        ]
        imputed_X = imputed_X.mean(axis=1)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(
            imputed_X,
            DATA["test_X_ori"],
            DATA["test_X_indicating_mask"],
        )
        logger.info(f"Lazy-loading GP-VAE test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
