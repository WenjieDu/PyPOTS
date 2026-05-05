"""
Test cases for the CLI command `pypots.cli.evaluate`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil
import tempfile
import unittest

import numpy as np
import pytest
from click.testing import CliRunner

from pypots.cli.evaluate import evaluate
from pypots.data.saving.h5 import save_dict_into_h5
from tests.cli.config import PROJECT_ROOT_DIR


class TestPyPOTSCLIEvaluate(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        n_samples, n_steps, n_features = 50, 6, 5
        np.random.seed(2023)
        predictions = np.random.randn(n_samples, n_steps, n_features)
        ground_truth = np.random.randn(n_samples, n_steps, n_features)
        indicating_mask = np.ones_like(ground_truth)

        self.pred_path = os.path.join(self.temp_dir, "predictions.h5")
        self.gt_path = os.path.join(self.temp_dir, "ground_truth.h5")

        save_dict_into_h5({"imputation": predictions}, self.pred_path)
        save_dict_into_h5({"X_ori": ground_truth, "indicating_mask": indicating_mask}, self.gt_path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-evaluate")
    def test_0_evaluate_imputation(self):
        runner = CliRunner()
        result = runner.invoke(
            evaluate,
            [
                "--predictions",
                self.pred_path,
                "--ground_truth",
                self.gt_path,
                "--task",
                "imputation",
                "--metrics",
                "mse,mae",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-evaluate")
    def test_1_evaluate_with_output(self):
        output_path = os.path.join(self.temp_dir, "eval_results.json")
        runner = CliRunner()
        result = runner.invoke(
            evaluate,
            [
                "--predictions",
                self.pred_path,
                "--ground_truth",
                self.gt_path,
                "--task",
                "imputation",
                "--metrics",
                "mse,mae",
                "--output",
                output_path,
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(output_path)


if __name__ == "__main__":
    unittest.main()
