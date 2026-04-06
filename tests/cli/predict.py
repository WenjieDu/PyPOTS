"""
Test cases for the CLI command `pypots.cli.predict`.
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

from pypots.cli.predict import predict
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import (
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    N_STEPS,
    N_FEATURES,
    N_PRED_STEPS,
)


class TestPyPOTSCLIPredict(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
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
        runner = CliRunner()
        result = runner.invoke(
            predict,
            [
                "--model_path",
                self.model_save_path,
                "--test_set",
                GENERAL_H5_TEST_SET_PATH,
                "--config",
                self.config_path,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-predict")
    def test_1_predict_with_output(self):
        output_path = os.path.join(self.temp_dir, "predictions.h5")
        runner = CliRunner()
        result = runner.invoke(
            predict,
            [
                "--model_path",
                self.model_save_path,
                "--test_set",
                GENERAL_H5_TEST_SET_PATH,
                "--config",
                self.config_path,
                "--output",
                output_path,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(output_path)


if __name__ == "__main__":
    unittest.main()
