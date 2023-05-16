"""
Test cases for data classes with the lazy-loading strategy of reading from files.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import h5py
import numpy as np
import pytest

from pypots.imputation import SAITS
from pypots.optim import Adam, AdamW, Adagrad, SGD, RMSprop
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae
from tests.global_test_config import DATA

TRAIN_SET = {"X": DATA["train_X"]}
VAL_SET = {
    "X": DATA["val_X"],
    "X_intact": DATA["val_X_intact"],
    "indicating_mask": DATA["val_X_indicating_mask"],
}
TEST_SET = {"X": DATA["test_X"]}


EPOCHS = 3


def save_data_set_into_h5(data, path):
    with h5py.File(path, "w") as hf:
        for i in data.keys():
            tp = int if i == "y" else "float32"
            hf.create_dataset(i, data=data[i].astype(tp))


class TestAdam(unittest.TestCase):
    logger.info("Running tests for Adam...")

    # initialize an Adam optimizer
    adam = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        d_model=128,
        d_inner=64,
        n_heads=2,
        d_k=64,
        d_v=64,
        dropout=0.1,
        optimizer=adam,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="optim-adam")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")


class TestAdamW(unittest.TestCase):
    logger.info("Running tests for AdamW...")

    # initialize an Adam optimizer
    adamw = AdamW(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        d_model=128,
        d_inner=64,
        n_heads=2,
        d_k=64,
        d_v=64,
        dropout=0.1,
        optimizer=adamw,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="optim-adamw")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")


class TestAdagrad(unittest.TestCase):
    logger.info("Running tests for Adagrad...")

    # initialize an Adam optimizer
    adagrad = Adagrad(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        d_model=128,
        d_inner=64,
        n_heads=2,
        d_k=64,
        d_v=64,
        dropout=0.1,
        optimizer=adagrad,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="optim-adagrad")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")


class TestSGD(unittest.TestCase):
    logger.info("Running tests for SGD...")

    # initialize an Adam optimizer
    sgd = SGD(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        d_model=128,
        d_inner=64,
        n_heads=2,
        d_k=64,
        d_v=64,
        dropout=0.1,
        optimizer=sgd,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="optim-sgd")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")


class TestRMSprop(unittest.TestCase):
    logger.info("Running tests for RMSprop...")

    # initialize an Adam optimizer
    rmsprop = RMSprop(lr=0.001, weight_decay=1e-5)

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=1,
        d_model=128,
        d_inner=64,
        n_heads=2,
        d_k=64,
        d_v=64,
        dropout=0.1,
        optimizer=rmsprop,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="optim-rmsprop")
    def test_0_fit(self):
        self.saits.fit(TRAIN_SET, VAL_SET)
        imputed_X = self.saits.impute(TEST_SET)
        assert not np.isnan(
            imputed_X
        ).any(), "Output still has missing values after running impute()."
        test_MAE = cal_mae(
            imputed_X, DATA["test_X_intact"], DATA["test_X_indicating_mask"]
        )
        logger.info(f"SAITS test_MAE: {test_MAE}")


if __name__ == "__main__":
    unittest.main()
