"""
Test cases for the functions and classes in package `pypots.cli.evaluate`.
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

from pypots.cli.evaluate import evaluate_command_factory
from pypots.data.saving.h5 import save_dict_into_h5
from tests.cli.config import PROJECT_ROOT_DIR


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIEvaluate(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "predictions": None,
        "ground_truth": None,
        "task": None,
        "metrics": None,
        "output": None,
    }
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)
        # create mock prediction and ground truth H5 files
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
        arguments = copy(self.default_arguments)
        arguments["predictions"] = self.pred_path
        arguments["ground_truth"] = self.gt_path
        arguments["task"] = "imputation"
        arguments["metrics"] = "mse,mae"
        args = Namespace(**arguments)
        evaluate_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-evaluate")
    def test_1_evaluate_with_output(self):
        arguments = copy(self.default_arguments)
        arguments["predictions"] = self.pred_path
        arguments["ground_truth"] = self.gt_path
        arguments["task"] = "imputation"
        arguments["metrics"] = "mse,mae"
        arguments["output"] = os.path.join(self.temp_dir, "eval_results.json")
        args = Namespace(**arguments)
        evaluate_command_factory(args).run()
        assert os.path.exists(os.path.join(self.temp_dir, "eval_results.json"))


if __name__ == "__main__":
    unittest.main()
