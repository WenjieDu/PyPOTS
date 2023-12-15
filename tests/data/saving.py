"""
Test cases for data saving utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest

import pytest

from pypots.data.saving import save_dict_into_h5, pickle_dump, pickle_load
from pypots.utils.logging import logger


class TestLazyLoadingClasses(unittest.TestCase):
    logger.info("Running tests for data saving utils...")

    data_to_save = {
        "a": 1,
        "b": 2,
        "c": {
            "d": 0,
        },
    }

    @pytest.mark.xdist_group(name="data-saving-h5")
    def test_0_save_dict_into_h5(self):
        save_dict_into_h5(self.data_to_save, "tests/data/saving_with_h5.h5")

    @pytest.mark.xdist_group(name="data-saving-pickle")
    def test_0_pickle_dump_load(self):
        pickle_dump(self.data_to_save, "tests/data/saving_with_pickle.pkl")
        loaded_data = pickle_load("tests/data/saving_with_pickle.pkl")
        assert loaded_data == self.data_to_save
