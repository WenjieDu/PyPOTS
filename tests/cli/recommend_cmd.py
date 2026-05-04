"""
Test cases for the CLI command `pypots.cli.recommend`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from pypots.cli.recommend import recommend
from pypots.data.saving.h5 import save_dict_into_h5
from tests.cli.config import PROJECT_ROOT_DIR


class TestPyPOTSCLIRecommend(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        self.runner = CliRunner()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-recommend")
    def test_0_recommend_with_dimensions(self):
        """Test recommend with explicit data dimensions."""
        result = self.runner.invoke(
            recommend,
            [
                "--task",
                "imputation",
                "--model",
                "SAITS",
                "--n_steps",
                "24",
                "--n_features",
                "10",
                "--n_samples",
                "500",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "Recommended Configuration" in result.output
        assert "n_steps: 24" in result.output
        assert "n_features: 10" in result.output

    @pytest.mark.xdist_group(name="cli-recommend")
    def test_1_recommend_from_h5(self):
        """Test recommend with an H5 data file."""
        np.random.seed(42)
        h5_path = os.path.join(self.temp_dir, "train.h5")
        X = np.random.randn(100, 16, 8)
        X[np.random.rand(*X.shape) < 0.15] = np.nan
        save_dict_into_h5({"X": X}, h5_path)

        output_yaml = os.path.join(self.temp_dir, "config.yaml")
        result = self.runner.invoke(
            recommend,
            [
                "--task",
                "imputation",
                "--model",
                "SAITS",
                "--data",
                h5_path,
                "--output",
                output_yaml,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(output_yaml)
        assert "Config saved to" in result.output

        # verify the YAML file
        import yaml

        with open(output_yaml) as f:
            config = yaml.safe_load(f)
        assert config["task"] == "imputation"
        assert config["model"]["name"] == "SAITS"
        assert config["model"]["n_steps"] == 16
        assert config["model"]["n_features"] == 8

    @pytest.mark.xdist_group(name="cli-recommend")
    def test_2_recommend_from_csv(self):
        """Test recommend with a CSV data file."""
        csv_path = os.path.join(self.temp_dir, "data.csv")
        rows = []
        for sid in range(15):
            for step in range(12):
                rows.append(
                    {
                        "SAMPLE_ID": sid,
                        "feat_0": np.random.randn(),
                        "feat_1": np.random.randn(),
                        "feat_2": np.random.randn(),
                        "CLAF_TARGET": sid % 2,
                    }
                )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        result = self.runner.invoke(
            recommend,
            ["--task", "classification", "--model", "TimesNet", "--data", csv_path],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "n_classes" in result.output or "n_classes:" in result.output

    @pytest.mark.xdist_group(name="cli-recommend")
    def test_3_recommend_default_model(self):
        """Test recommend uses default model when --model is not specified."""
        result = self.runner.invoke(
            recommend,
            ["--task", "imputation", "--n_steps", "48", "--n_features", "35"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "SAITS" in result.output

    @pytest.mark.xdist_group(name="cli-recommend")
    def test_4_recommend_all_models(self):
        """Test recommend works for all supported model types."""
        models_tasks = [
            ("SAITS", "imputation"),
            ("TimesNet", "classification"),
            ("TEFN", "forecasting"),
            ("CRLI", "clustering"),
            ("TimeMixer", "anomaly_detection"),
        ]
        for model, task in models_tasks:
            result = self.runner.invoke(
                recommend,
                [
                    "--task",
                    task,
                    "--model",
                    model,
                    "--n_steps",
                    "24",
                    "--n_features",
                    "10",
                    "--n_samples",
                    "200",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, f"Failed for {model}/{task}: {result.output}"


if __name__ == "__main__":
    unittest.main()
