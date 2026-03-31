"""
Test cases for the functions and classes in package `pypots.cli.benchmark`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from copy import copy

import pytest
import yaml

from pypots.cli.benchmark import benchmark_command_factory
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import (
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    N_STEPS,
    N_FEATURES,
    N_PRED_STEPS,
)


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIBenchmark(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "config": None,
        "device": None,
        "seed": None,
        "output": None,
    }
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        self.config_path = os.path.join(self.temp_dir, "benchmark_config.yaml")
        config = {
            "task": "imputation",
            "models": [
                {
                    "name": "Mean",
                    "params": {
                        "n_steps": N_STEPS + N_PRED_STEPS,
                        "n_features": N_FEATURES,
                    },
                },
                {
                    "name": "Median",
                    "params": {
                        "n_steps": N_STEPS + N_PRED_STEPS,
                        "n_features": N_FEATURES,
                    },
                },
            ],
            "data": {
                "train_set": GENERAL_H5_TRAIN_SET_PATH,
                "val_set": GENERAL_H5_VAL_SET_PATH,
                "test_set": GENERAL_H5_TEST_SET_PATH,
            },
            "metrics": ["mse", "mae"],
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-benchmark")
    def test_0_benchmark(self):
        arguments = copy(self.default_arguments)
        arguments["config"] = self.config_path
        arguments["seed"] = 2023
        args = Namespace(**arguments)
        benchmark_command_factory(args).run()


if __name__ == "__main__":
    unittest.main()
