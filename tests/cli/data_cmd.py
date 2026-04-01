"""
Test cases for the CLI command `pypots.cli.data`.
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

from pypots.cli.data import data
from pypots.data.saving.h5 import save_dict_into_h5
from tests.cli.config import PROJECT_ROOT_DIR
from tests.global_test_config import GENERAL_H5_TRAIN_SET_PATH


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIData(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-data")
    def test_0_describe(self):
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            data, ["describe", "--input", GENERAL_H5_TRAIN_SET_PATH],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_1_split(self):
        n_samples, n_steps, n_features = 100, 6, 5
        np.random.seed(2023)
        input_data = {"X": np.random.randn(n_samples, n_steps, n_features)}
        input_path = os.path.join(self.temp_dir, "data_to_split.h5")
        save_dict_into_h5(input_data, input_path)

        split_output_dir = os.path.join(self.temp_dir, "split_output")
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            data,
            ["split", "--input", input_path, "--output_dir", split_output_dir,
             "--train_ratio", "0.7", "--val_ratio", "0.1", "--test_ratio", "0.2"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_2_convert(self):
        np.random.seed(2023)
        npy_path = os.path.join(self.temp_dir, "data.npy")
        np.save(npy_path, np.random.randn(50, 6, 5))

        output_path = os.path.join(self.temp_dir, "converted.h5")
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            data,
            ["convert", "--input", npy_path, "--output", output_path],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_3_list(self):
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(data, ["list"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_4_load(self):
        output_dir = os.path.join(self.temp_dir, "benchmark_data")
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            data,
            ["load", "--dataset", "physionet_2012", "--output_dir", output_dir,
             "--subset", "set-a", "--rate", "0.1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(os.path.join(output_dir, "train.h5")), "train.h5 should exist"
        assert os.path.exists(os.path.join(output_dir, "val.h5")), "val.h5 should exist"
        assert os.path.exists(os.path.join(output_dir, "test.h5")), "test.h5 should exist"


if __name__ == "__main__":
    unittest.main()
