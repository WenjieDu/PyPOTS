"""
Test cases for the functions and classes in package `pypots.cli.predict`.
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

from pypots.cli.predict import predict_command_factory
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
class TestPyPOTSCLIPredict(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "model_path": None,
        "test_set": None,
        "task": None,
        "model": None,
        "config": None,
        "output": None,
        "device": None,
        "file_type": "hdf5",
    }
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        # Write a config file matching the model architecture
        self.config_path = os.path.join(self.temp_dir, "saits_config.yaml")
        config = {
            "task": "imputation",
            "model": {
                "name": "SAITS",
                "n_steps": N_STEPS + N_PRED_STEPS,
                "n_features": N_FEATURES,
                "n_layers": 1,
                "d_model": 8,
                "n_heads": 1,
                "d_k": 8,
                "d_v": 8,
                "d_ffn": 8,
            },
            "training": {
                "epochs": 1,
            },
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Train a minimal SAITS model and save it
        from pypots.imputation import SAITS
        self.model = SAITS(
            n_steps=N_STEPS + N_PRED_STEPS,
            n_features=N_FEATURES,
            n_layers=1,
            d_model=8,
            n_heads=1,
            d_k=8,
            d_v=8,
            d_ffn=8,
            epochs=1,
        )
        self.model.fit(train_set=GENERAL_H5_TRAIN_SET_PATH, val_set=GENERAL_H5_VAL_SET_PATH)
        self.model_save_path = os.path.join(self.temp_dir, "saits_model.pypots")
        self.model.save(self.model_save_path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-predict")
    def test_0_predict(self):
        arguments = copy(self.default_arguments)
        arguments["config"] = self.config_path
        arguments["model_path"] = self.model_save_path
        arguments["test_set"] = GENERAL_H5_TEST_SET_PATH
        args = Namespace(**arguments)
        predict_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-predict")
    def test_1_predict_with_output(self):
        arguments = copy(self.default_arguments)
        arguments["config"] = self.config_path
        arguments["model_path"] = self.model_save_path
        arguments["test_set"] = GENERAL_H5_TEST_SET_PATH
        arguments["output"] = os.path.join(self.temp_dir, "predictions.h5")
        args = Namespace(**arguments)
        predict_command_factory(args).run()
        assert os.path.exists(os.path.join(self.temp_dir, "predictions.h5"))


if __name__ == "__main__":
    unittest.main()
