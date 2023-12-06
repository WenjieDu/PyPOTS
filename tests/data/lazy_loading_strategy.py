"""
Test cases for data classes with the lazy-loading strategy of reading from files.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest

from pypots.classification import BRITS, GRUD
from pypots.data.saving import save_dict_into_h5
from pypots.imputation import SAITS
from pypots.utils.logging import logger
from tests.global_test_config import DATA, DATA_SAVING_DIR

TRAIN_SET_NAME = "train_set.h5"
TRAIN_SET_PATH = f"{DATA_SAVING_DIR}/{TRAIN_SET_NAME}"
VAL_SET_NAME = "val_set.h5"
VAL_SET_PATH = f"{DATA_SAVING_DIR}/{VAL_SET_NAME}"
TEST_SET_NAME = "test_set.h5"
TEST_SET_PATH = f"{DATA_SAVING_DIR}/{TEST_SET_NAME}"
IMPUTATION_TRAIN_SET_NAME = "imputation_train_set.h5"
IMPUTATION_TRAIN_SET_PATH = f"{DATA_SAVING_DIR}/{IMPUTATION_TRAIN_SET_NAME}"
IMPUTATION_VAL_SET_NAME = "imputation_val_set.h5"
IMPUTATION_VAL_SET_PATH = f"{DATA_SAVING_DIR}/{IMPUTATION_VAL_SET_NAME}"

EPOCHS = 1


class TestLazyLoadingClasses(unittest.TestCase):
    logger.info("Running tests for Dataset classes with lazy-loading strategy...")

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
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
        epochs=EPOCHS,
    )

    # initialize a BRITS model for testing DatasetForBRITS
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        rnn_hidden_size=256,
        epochs=EPOCHS,
    )

    # initialize a GRUD model for testing DatasetForGRUD
    grud = GRUD(
        DATA["n_steps"],
        DATA["n_features"],
        n_classes=DATA["n_classes"],
        rnn_hidden_size=256,
        epochs=EPOCHS,
    )

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_0_save_datasets_into_files(self):
        # create the dir for saving files
        os.makedirs(DATA_SAVING_DIR, exist_ok=True)

        if not os.path.exists(TRAIN_SET_PATH):
            save_dict_into_h5(
                {"X": DATA["train_X"], "y": DATA["train_y"].astype(float)},
                DATA_SAVING_DIR,
                TRAIN_SET_NAME,
            )

        if not os.path.exists(VAL_SET_PATH):
            save_dict_into_h5(
                {"X": DATA["val_X"], "y": DATA["val_y"].astype(float)},
                DATA_SAVING_DIR,
                VAL_SET_NAME,
            )

        if not os.path.exists(IMPUTATION_TRAIN_SET_PATH):
            save_dict_into_h5(
                {"X": DATA["train_X"]}, DATA_SAVING_DIR, IMPUTATION_TRAIN_SET_NAME
            )

        if not os.path.exists(IMPUTATION_VAL_SET_PATH):
            save_dict_into_h5(
                {
                    "X": DATA["val_X"],
                    "X_intact": DATA["val_X_intact"],
                    "indicating_mask": DATA["val_X_indicating_mask"],
                },
                DATA_SAVING_DIR,
                IMPUTATION_VAL_SET_NAME,
            )

        if not os.path.exists(TEST_SET_PATH):
            save_dict_into_h5(
                {
                    "X": DATA["test_X"],
                    "X_intact": DATA["test_X_intact"],
                    "indicating_mask": DATA["test_X_indicating_mask"],
                },
                DATA_SAVING_DIR,
                TEST_SET_NAME,
            )

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_1_DatasetForMIT_BaseDataset(self):
        self.saits.fit(
            train_set=IMPUTATION_TRAIN_SET_PATH, val_set=IMPUTATION_VAL_SET_PATH
        )
        _ = self.saits.impute(X=TEST_SET_PATH)

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_2_DatasetForBRITS(self):
        self.brits.fit(train_set=TRAIN_SET_PATH, val_set=VAL_SET_PATH)
        _ = self.brits.classify(X=TEST_SET_PATH)

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_3_DatasetForGRUD(self):
        self.grud.fit(train_set=TRAIN_SET_PATH, val_set=VAL_SET_PATH)
        _ = self.grud.classify(X=TEST_SET_PATH)


if __name__ == "__main__":
    unittest.main()
