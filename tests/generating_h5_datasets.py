"""
Test cases for data saving utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
import unittest

import pytest

from tests.global_test_config import (
    DATA,
    H5_TRAIN_SET_PATH,
    H5_VAL_SET_PATH,
    H5_TEST_SET_PATH,
)
from pypots.data.saving import save_dict_into_h5
from pypots.utils.logging import logger


class TestGeneratingH5Datasets(unittest.TestCase):
    logger.info("Generating HDF5 data files...")

    @pytest.mark.xdist_group(name="generating-h5-datasets")
    def test_0_save_data_into_h5(self):
        if not os.path.exists(H5_TRAIN_SET_PATH):
            save_dict_into_h5(
                {
                    "X": DATA["train_X"],
                    "y": DATA["train_y"].astype(float),
                },
                H5_TRAIN_SET_PATH,
            )

        if not os.path.exists(H5_VAL_SET_PATH):
            save_dict_into_h5(
                {
                    "X": DATA["val_X"],
                    "X_intact": DATA["val_X_intact"],
                    "y": DATA["val_y"].astype(float),
                },
                H5_VAL_SET_PATH,
            )

        if not os.path.exists(H5_TEST_SET_PATH):
            save_dict_into_h5(
                {
                    "X": DATA["test_X"],
                    "X_intact": DATA["test_X_intact"],
                    "y": DATA["test_y"].astype(float),
                },
                H5_TEST_SET_PATH,
            )
