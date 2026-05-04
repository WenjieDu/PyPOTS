"""
Test cases for the CLI command `pypots.cli.benchmark`.
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

from pypots.cli.benchmark import benchmark
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import (
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    N_STEPS,
    N_FEATURES,
    N_PRED_STEPS,
)


class TestPyPOTSCLIBenchmark(unittest.TestCase):
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
        runner = CliRunner()
        result = runner.invoke(
            benchmark,
            ["--config", self.config_path, "--seed", "2023"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
