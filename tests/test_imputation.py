"""
Test cases for imputation models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    MRNN,
    LOCF,
)
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae
from tests.global_test_config import (
    DATA,
    RESULT_SAVING_DIR,
    check_tb_and_model_checkpoints_existence,
)

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

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=EPOCH,
        saving_path=saving_path,
        optimizer=optimizer,
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

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.saits)

        # save the trained model into file, and check if the path exists
        self.saits.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.saits.load_model(saved_model_path)


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
        epochs=EPOCH,
        saving_path=saving_path,
        optimizer=optimizer,
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

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.transformer)

        # save the trained model into file, and check if the path exists
        self.transformer.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.transformer.load_model(saved_model_path)


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
        epochs=EPOCH,
        saving_path=f"{RESULT_SAVING_DIR_FOR_IMPUTATION}/BRITS",
        optimizer=optimizer,
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


class TestMRNN(unittest.TestCase):
    logger.info("Running tests for an imputation model MRNN...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "MRNN")
    model_save_name = "saved_MRNN_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a MRNN model
    mrnn = MRNN(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        epochs=EPOCH,
        saving_path=f"{RESULT_SAVING_DIR_FOR_IMPUTATION}/MRNN",
        optimizer=optimizer,
    )

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_0_fit(self):
        self.mrnn.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_1_impute(self):
        imputed_X = self.mrnn.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"MRNN test_MAE: {test_MAE}")

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_2_parameters(self):
        assert hasattr(self.mrnn, "model") and self.mrnn.model is not None

        assert hasattr(self.mrnn, "optimizer") and self.mrnn.optimizer is not None

        assert hasattr(self.mrnn, "best_loss")
        self.assertNotEqual(self.mrnn.best_loss, float("inf"))

        assert (
            hasattr(self.mrnn, "best_model_dict")
            and self.mrnn.best_model_dict is not None
        )

    @pytest.mark.xdist_group(name="imputation-mrnn")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(
            self.saving_path
        ), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.mrnn)

        # save the trained model into file, and check if the path exists
        self.mrnn.save_model(
            saving_dir=self.saving_path, file_name=self.model_save_name
        )

        # test loading the saved model, not necessary, but need to test
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.mrnn.load_model(saved_model_path)


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
