"""
Test cases for imputation models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import unittest

import numpy as np

from pypots.imputation import (
    SAITS,
    Transformer,
    BRITS,
    LOCF,
)
from pypots.tests.unified_data_for_test import DATA
from pypots.utils.metrics import cal_mae
from pypots.utils.logging import logger

EPOCH = 5

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {"X": DATA["val_X"]}
TEST_SET = {"X": DATA["test_X"]}


class TestSAITS(unittest.TestCase):
    logger.info("Running tests for an imputation model SAITS...")

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
    )

    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)

    def test_1_impute(self):
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")

    def test_2_parameters(self):
        assert hasattr(self.saits, "model") and self.saits.model is not None

        assert hasattr(self.saits, "optimizer") and self.saits.optimizer is not None

        assert hasattr(self.saits, "best_loss")
        self.assertNotEqual(self.saits.best_loss, float("inf"))

        assert (
            hasattr(self.saits, "best_model_dict")
            and self.saits.best_model_dict is not None
        )


class TestTransformer(unittest.TestCase):
    logger.info("Running tests for an imputation model Transformer...")

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
    )

    def test_0_fit(self):
        self.transformer.fit(TRAIN_SET, VAL_SET)

    def test_1_impute(self):
        imputed_X = self.transformer.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"Transformer test_MAE: {test_MAE}")

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


class TestBRITS(unittest.TestCase):
    logger.info("Running tests for an imputation model BRITS...")

    # initialize a BRITS model
    brits = BRITS(DATA["n_steps"], DATA["n_features"], 256, epochs=EPOCH)

    def test_0_fit(self):
        self.brits.fit(TRAIN_SET, VAL_SET)

    def test_1_impute(self):
        imputed_X = self.brits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"BRITS test_MAE: {test_MAE}")

    def test_2_parameters(self):
        assert hasattr(self.brits, "model") and self.brits.model is not None

        assert hasattr(self.brits, "optimizer") and self.brits.optimizer is not None

        assert hasattr(self.brits, "best_loss")
        self.assertNotEqual(self.brits.best_loss, float("inf"))

        assert (
            hasattr(self.brits, "best_model_dict")
            and self.brits.best_model_dict is not None
        )


class TestLOCF(unittest.TestCase):
    logger.info("Running tests for an imputation model LOCF...")
    locf = LOCF(nan=0)

    def test_0_impute(self):
        test_X_imputed = self.locf.impute(TEST_SET)
        assert not np.isnan(
            test_X_imputed
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            test_X_imputed, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"LOCF test_MAE: {test_MAE}")

    def test_1_parameters(self):
        assert hasattr(self.locf, "nan") and self.locf.nan is not None


if __name__ == "__main__":
    unittest.main()
