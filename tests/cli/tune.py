"""
Test cases for the functions and classes in package `pypots.cli.tune`.
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

from pypots.cli.tune import tune_command_factory
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import (
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    N_STEPS,
    N_FEATURES,
    N_PRED_STEPS,
)


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLITune(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "config": None,
        "task": None,
        "model": None,
        "n_trials": None,
        "device": None,
    }
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        self.config_path = os.path.join(self.temp_dir, "tune_config.yaml")
        config = {
            "task": "imputation",
            "model": {
                "name": "SAITS",
                "n_steps": N_STEPS + N_PRED_STEPS,
                "n_features": N_FEATURES,
                "n_heads": 1,
                "d_k": 8,
                "d_v": 8,
                "d_ffn": 32,
            },
            "search_space": {
                "n_layers": {"type": "int", "low": 1, "high": 2},
                "d_model": {"type": "categorical", "choices": [32, 64]},
            },
            "tuner": {
                "sampler": "TPE",
                "n_trials": 2,
                "direction": "minimize",
            },
            "training": {
                "epochs": 1,
            },
            "data": {
                "train_set": GENERAL_H5_TRAIN_SET_PATH,
                "val_set": GENERAL_H5_VAL_SET_PATH,
            },
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-tune")
    def test_0_tune(self):
        arguments = copy(self.default_arguments)
        arguments["config"] = self.config_path
        args = Namespace(**arguments)
        tune_command_factory(args).run()


if __name__ == "__main__":
    unittest.main()
