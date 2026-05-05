"""
Test cases for the CLI command `pypots.cli.tune`.
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

from pypots.cli.tune import tune
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import (
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    N_STEPS,
    N_FEATURES,
    N_PRED_STEPS,
)


class TestPyPOTSCLITune(unittest.TestCase):
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
                "n_trials": 20,
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
        runner = CliRunner()
        result = runner.invoke(tune, ["--config", self.config_path], catch_exceptions=False)
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
