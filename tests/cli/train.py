"""
Test cases for the CLI command `pypots.cli.train`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil
import tempfile
import unittest

import pytest
import yaml
from click.testing import CliRunner

from pypots.cli.train import train
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import (
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    N_STEPS,
    N_FEATURES,
    N_PRED_STEPS,
)


class TestPyPOTSCLITrain(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        self.config_path = os.path.join(self.temp_dir, "train_config.yaml")
        config = {
            "task": "imputation",
            "model": {
                "name": "Mean",
                "n_steps": N_STEPS + N_PRED_STEPS,
                "n_features": N_FEATURES,
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
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

    @pytest.mark.xdist_group(name="cli-train")
    def test_0_train(self):
        runner = CliRunner()
        result = runner.invoke(train, ["--config", self.config_path], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-train")
    def test_1_train_with_overrides(self):
        runner = CliRunner()
        result = runner.invoke(train, ["--config", self.config_path, "--epochs", "1"], catch_exceptions=False)
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
