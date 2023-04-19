"""
Test cases for imputation models.
"""
import os.path

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import unittest

import numpy as np
import pytest

from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    LOCF,
)
from pypots.tests.global_test_config import DATA, RESULT_SAVING_DIR
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae

EPOCH = 5

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {
    "X": DATA["val_X"],
    "X_intact": DATA["val_X_intact"],
    "indicating_mask": DATA["val_X_indicating_mask"],
}
TEST_SET = {"X": DATA["test_X"]}

RESULT_SAVING_DIR_FOR_IMPUTATION = os.path.join(RESULT_SAVING_DIR, "imputation")


class TestSAITS(unittest.TestCase):
    logger.info("Running tests for an imputation model SAITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "SAITS")
    model_save_name = "saved_saits_model.pypots"

    # initialize a SAITS model
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=EPOCH,
        tb_file_saving_path=saving_path,
    )

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_1_impute(self):
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_2_parameters(self):
        assert hasattr(self.saits, "model") and self.saits.model is not None

        assert hasattr(self.saits, "optimizer") and self.saits.optimizer is not None

        assert hasattr(self.saits, "best_loss")
        self.assertNotEqual(self.saits.best_loss, float("inf"))

        assert (
            hasattr(self.saits, "best_model_dict")
            and self.saits.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-saits")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # whether the tensorboard file exists
        assert (
            self.saits.tb_file_saving_path is not None
            and len(os.listdir(self.saits.tb_file_saving_path)) > 0
        ), "tensorboard file does not exist"

        # save the trained model into file, and check if the path exists
        self.saits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


class TestTransformer(unittest.TestCase):
    logger.info("Running tests for an imputation model Transformer...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "Transformer")
    model_save_name = "saved_transformer_model.pypots"

    # initialize a Transformer model
    transformer = Transformer(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=EPOCH,
        tb_file_saving_path=saving_path,
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
        test_MAE = cal_mae(
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

        # whether the tensorboard file exists
        assert (
                self.transformer.tb_file_saving_path is not None
                and len(os.listdir(self.transformer.tb_file_saving_path)) > 0
        ), "tensorboard file does not exist"

        # save the trained model into file, and check if the path exists
        self.transformer.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


class TestBRITS(unittest.TestCase):
    logger.info("Running tests for an imputation model BRITS...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "BRITS")
    model_save_name = "saved_BRITS_model.pypots"

    # initialize a BRITS model
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCH,
        tb_file_saving_path=f"{RESULT_SAVING_DIR_FOR_IMPUTATION}/BRITS",
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

        # whether the tensorboard file exists
        assert (
                self.brits.tb_file_saving_path is not None
                and len(os.listdir(self.brits.tb_file_saving_path)) > 0
        ), "tensorboard file does not exist"

        # save the trained model into file, and check if the path exists
        self.brits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        assert os.path.exists(
            saved_model_path
        ), f"file {self.saving_path} does not exist, model not saved"


class TestLOCF(unittest.TestCase):
    logger.info("Running tests for an imputation model LOCF...")
    locf = LOCF(nan=0)

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_0_impute(self):
        test_X_imputed = self.locf.impute(TEST_SET)
        assert not np.isnan(
            test_X_imputed
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"LOCF test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-locf")
    def test_1_parameters(self):
        assert hasattr(self.locf, "nan") and self.locf.nan is not None


if __name__ == "__main__":
    unittest.main()
