"""
Test cases for data classes with the lazy-loading strategy of reading from files.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
import unittest

import h5py
import pytest

from pypots.classification import BRITS, GRUD
from pypots.imputation import SAITS
from pypots.tests.unified_data_for_test import DATA
from pypots.utils.logging import logger

TRAIN_SET = "./train_set.h5"
VAL_SET = "./val_set.h5"
TEST_SET = "./test_set.h5"

IMPUTATION_TRAIN_SET = "./imputation_train_set.h5"
IMPUTATION_VAL_SET = "./imputation_val_set.h5"


def save_data_set_into_h5(data, path):
    with h5py.File(path, "w") as hf:
        for i in data.keys():
            tp = int if i == "y" else "float32"
            hf.create_dataset(i, data=data[i].astype(tp))


EPOCHS = 1

save_data_set_into_h5(
    {"X": DATA["train_X"], "y": DATA["train_y"].astype(int)}, TRAIN_SET
)
save_data_set_into_h5({"X": DATA["val_X"], "y": DATA["val_y"].astype(int)}, VAL_SET)
save_data_set_into_h5({"X": DATA["train_X"]}, IMPUTATION_TRAIN_SET)
save_data_set_into_h5({"X": DATA["val_X"]}, IMPUTATION_VAL_SET)

save_data_set_into_h5(
    {
        "X": DATA["test_X"],
        "X_intact": DATA["test_X_intact"],
        "X_indicating_mask": DATA["test_X_indicating_mask"],
    },
    TEST_SET,
)


class TestLazyLoadingClasses(unittest.TestCase):
    logger.info("Running tests for Dataset classes with lazy-loading strategy...")

    # initialize a SAITS model for testing DatasetForMIT and BaseDataset
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
        epochs=EPOCHS,
    )

    # initialize a BRITS model for testing DatasetForBRITS
    brits = BRITS(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        n_classes=DATA["n_classes"],
        epochs=EPOCHS,
    )

    # initialize a GRUD model for testing DatasetForGRUD
    grud = GRUD(
        DATA["n_steps"],
        DATA["n_features"],
        256,
        n_classes=DATA["n_classes"],
        epochs=EPOCHS,
    )

    def setUp(self) -> None:
        assert os.path.exists(TRAIN_SET)
        assert os.path.exists(VAL_SET)
        assert os.path.exists(TEST_SET)

        assert os.path.exists(IMPUTATION_TRAIN_SET)
        assert os.path.exists(IMPUTATION_VAL_SET)

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_0_DatasetForMIT(self):
        self.saits.fit(train_set=IMPUTATION_TRAIN_SET, val_set=IMPUTATION_VAL_SET)

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_1_BaseDataset(self):
        _ = self.saits.impute(X=TEST_SET)

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_2_DatasetForBRITS(self):
        self.brits.fit(train_set=TRAIN_SET, val_set=VAL_SET)
        _ = self.brits.classify(X=TEST_SET)

    @pytest.mark.xdist_group(name="data-lazy-loading")
    def test_3_DatasetForGRUD(self):
        self.grud.fit(train_set=TRAIN_SET, val_set=VAL_SET)
        _ = self.grud.classify(X=TEST_SET)


if __name__ == "__main__":
    unittest.main()
