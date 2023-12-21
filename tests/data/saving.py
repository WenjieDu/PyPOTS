"""
Test cases for data saving utils.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import unittest

import numpy as np
import pytest

from pypots.data.saving import (
    save_dict_into_h5,
    load_dict_from_h5,
    pickle_dump,
    pickle_load,
)
from pypots.utils.logging import logger


class TestLazyLoadingClasses(unittest.TestCase):
    logger.info("Running tests for data saving utils...")

    data_to_save = {
        "a": 1,
        "b": 2,
        "c": {
            "d": 0,
            "e": {
                "f": np.random.randn(10, 10),
            },
        },
    }

    h5_saving_path = "tests/data/saving_with_h5.h5"
    pickle_saving_path = "tests/data/saving_with_pickle.pkl"

    @pytest.mark.xdist_group(name="data-saving-h5")
    def test_0_save_dict_into_h5(self):
        save_dict_into_h5(self.data_to_save, self.h5_saving_path)
        loaded_data = load_dict_from_h5(self.h5_saving_path)
        assert loaded_data["c"]["d"] == self.data_to_save["c"]["d"]
        assert (loaded_data["c"]["e"]["f"] == self.data_to_save["c"]["e"]["f"]).all()

    @pytest.mark.xdist_group(name="data-saving-pickle")
    def test_0_pickle_dump_load(self):
        pickle_dump(self.data_to_save, self.pickle_saving_path)
        loaded_data = pickle_load(self.pickle_saving_path)
        assert (loaded_data["c"]["e"]["f"] == self.data_to_save["c"]["e"]["f"]).all()
