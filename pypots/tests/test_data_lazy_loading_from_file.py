"""
Test cases for data classes with the lazy-loading strategy of reading from files.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
import unittest

from pypots.classification import BRITS, GRUD
from pypots.imputation import SAITS
from pypots.tests.unified_data_for_test import (
    DATA,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    IMPUTATION_TRAIN_SET,
    IMPUTATION_VAL_SET,
)

EPOCHS = 1


class TestLazyLoadingClasses(unittest.TestCase):
    def setUp(self) -> None:

        assert os.path.exists(TRAIN_SET)
        assert os.path.exists(VAL_SET)
        assert os.path.exists(TEST_SET)

        assert os.path.exists(IMPUTATION_TRAIN_SET)
        assert os.path.exists(IMPUTATION_VAL_SET)

        self.saits = SAITS(
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

        self.brits = BRITS(
            DATA["n_steps"],
            DATA["n_features"],
            256,
            n_classes=DATA["n_classes"],
            epochs=EPOCHS,
        )

        self.grud = GRUD(
            DATA["n_steps"],
            DATA["n_features"],
            256,
            n_classes=DATA["n_classes"],
            epochs=EPOCHS,
        )

    def test_DatasetForMIT(self):
        self.saits.fit(train_set=IMPUTATION_TRAIN_SET, val_set=IMPUTATION_VAL_SET)
        _ = self.saits.impute(X=TEST_SET)

    def test_DatasetForBRITS(self):
        self.brits.fit(train_set=TRAIN_SET, val_set=VAL_SET)
        _ = self.brits.classify(X=TEST_SET)

    def test_DatasetForGRUD(self):
        self.grud.fit(train_set=TRAIN_SET, val_set=VAL_SET)
        _ = self.grud.classify(X=TEST_SET)


if __name__ == "__main__":
    unittest.main()
