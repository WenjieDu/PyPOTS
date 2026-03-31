"""
Test cases for the functions and classes in package `pypots.cli.data`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from copy import copy

import numpy as np
import pytest

from pypots.cli.data import data_command_factory
from pypots.data.saving.h5 import save_dict_into_h5
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import GENERAL_H5_TRAIN_SET_PATH


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIData(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "action": None,
        "input": None,
        "output": None,
        "output_dir": None,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "seed": 2024,
    }
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-data")
    def test_0_describe(self):
        arguments = copy(self.default_arguments)
        arguments["action"] = "describe"
        arguments["input"] = GENERAL_H5_TRAIN_SET_PATH
        args = Namespace(**arguments)
        data_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-data")
    def test_1_split(self):
        # create a test H5 file for splitting
        n_samples, n_steps, n_features = 100, 6, 5
        np.random.seed(2023)
        data = {"X": np.random.randn(n_samples, n_steps, n_features)}
        input_path = os.path.join(self.temp_dir, "data_to_split.h5")
        save_dict_into_h5(data, input_path)

        split_output_dir = os.path.join(self.temp_dir, "split_output")
        arguments = copy(self.default_arguments)
        arguments["action"] = "split"
        arguments["input"] = input_path
        arguments["output_dir"] = split_output_dir
        arguments["train_ratio"] = 0.7
        arguments["val_ratio"] = 0.1
        arguments["test_ratio"] = 0.2
        args = Namespace(**arguments)
        data_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-data")
    def test_2_convert(self):
        # create a numpy file to convert to H5
        np.random.seed(2023)
        npy_path = os.path.join(self.temp_dir, "data.npy")
        np.save(npy_path, np.random.randn(50, 6, 5))

        output_path = os.path.join(self.temp_dir, "converted.h5")
        arguments = copy(self.default_arguments)
        arguments["action"] = "convert"
        arguments["input"] = npy_path
        arguments["output"] = output_path
        args = Namespace(**arguments)
        data_command_factory(args).run()


if __name__ == "__main__":
    unittest.main()
