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


class TestPyPOTSCLIData(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(dir=PROJECT_ROOT_DIR)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.xdist_group(name="cli-data")
    def test_0_describe(self):
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["describe", "--input", GENERAL_H5_TRAIN_SET_PATH],
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
        runner = CliRunner()
        result = runner.invoke(
            data,
            [
                "split",
                "--input",
                input_path,
                "--output_dir",
                split_output_dir,
                "--train_ratio",
                "0.7",
                "--val_ratio",
                "0.1",
                "--test_ratio",
                "0.2",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_2_convert(self):
        np.random.seed(2023)
        npy_path = os.path.join(self.temp_dir, "data.npy")
        np.save(npy_path, np.random.randn(50, 6, 5))

        output_path = os.path.join(self.temp_dir, "converted.h5")
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["convert", "--input", npy_path, "--output", output_path],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_3_list(self):
        runner = CliRunner()
        result = runner.invoke(data, ["list"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_4_load(self):
        output_dir = os.path.join(self.temp_dir, "benchmark_data")
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["load", "--dataset", "physionet_2012", "--output_dir", output_dir, "--subset", "set-a", "--rate", "0.1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(os.path.join(output_dir, "train.h5")), "train.h5 should exist"
        assert os.path.exists(os.path.join(output_dir, "val.h5")), "val.h5 should exist"
        assert os.path.exists(os.path.join(output_dir, "test.h5")), "test.h5 should exist"

    @pytest.mark.xdist_group(name="cli-data")
    def test_5_describe_csv(self):
        """Test data describe with a CSV file."""
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "test_data.csv")
        np.random.seed(42)
        n_samples, n_steps, n_features = 10, 8, 3
        rows = []
        for sid in range(n_samples):
            for step in range(n_steps):
                row = {"SAMPLE_ID": sid}
                for f in range(n_features):
                    val = np.random.randn()
                    row[f"feat_{f}"] = val if np.random.rand() > 0.15 else np.nan
                row["CLAF_TARGET"] = sid % 2
                rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(data, ["describe", "--input", csv_path], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "Samples:" in result.output
        assert "Time steps:" in result.output
        assert "Features:" in result.output
        assert "Missing rate:" in result.output

    @pytest.mark.xdist_group(name="cli-data")
    def test_6_describe_csv_json(self):
        """Test data describe with --json flag."""
        import json
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "test_json.csv")
        np.random.seed(42)
        rows = []
        for sid in range(5):
            for step in range(4):
                rows.append({"SAMPLE_ID": sid, "feat_0": np.random.randn(), "feat_1": np.random.randn()})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(data, ["describe", "--input", csv_path, "--json"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["n_samples"] == 5
        assert parsed["n_steps"] == 4
        assert parsed["n_features"] == 2
        assert parsed["format"] == "csv"

    @pytest.mark.xdist_group(name="cli-data")
    def test_7_prepare_single_file(self):
        """Test data prepare in single-file mode."""
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "rw_train.csv")
        np.random.seed(42)
        rows = []
        for sid in range(20):
            for step in range(10):
                row = {"SAMPLE_ID": sid}
                for f in range(4):
                    val = np.random.randn()
                    row[f"feat_{f}"] = val if np.random.rand() > 0.1 else np.nan
                row["CLAF_TARGET"] = sid % 3
                rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        h5_path = os.path.join(self.temp_dir, "train.h5")
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["prepare", "--input", csv_path, "--output", h5_path, "--task", "imputation", "--set_type", "train"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(h5_path)

        from pypots.data.saving.h5 import load_dict_from_h5

        loaded = load_dict_from_h5(h5_path)
        assert "X" in loaded
        assert "X_ori" in loaded
        assert "y" in loaded
        assert loaded["X"].shape == (20, 10, 4)
        assert loaded["y"].shape == (20,)
        # for train set, X and X_ori should be identical
        np.testing.assert_array_equal(loaded["X"], loaded["X_ori"])

    @pytest.mark.xdist_group(name="cli-data")
    def test_8_prepare_batch_mode(self):
        """Test data prepare in batch mode with train/val/test."""
        import pandas as pd

        def make_csv(path, n_samples, seed=42):
            np.random.seed(seed)
            rows = []
            for sid in range(n_samples):
                for step in range(8):
                    row = {"SAMPLE_ID": sid}
                    for f in range(3):
                        val = np.random.randn()
                        row[f"feat_{f}"] = val if np.random.rand() > 0.1 else np.nan
                    row["CLAF_TARGET"] = sid % 2
                    rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)

        train_csv = os.path.join(self.temp_dir, "train.csv")
        val_csv = os.path.join(self.temp_dir, "val.csv")
        test_csv = os.path.join(self.temp_dir, "test.csv")
        make_csv(train_csv, 30, seed=1)
        make_csv(val_csv, 10, seed=2)
        make_csv(test_csv, 10, seed=3)

        h5_dir = os.path.join(self.temp_dir, "h5_output")
        runner = CliRunner()
        result = runner.invoke(
            data,
            [
                "prepare",
                "--train",
                train_csv,
                "--val",
                val_csv,
                "--test",
                test_csv,
                "--output_dir",
                h5_dir,
                "--task",
                "imputation",
                "--missing_rate",
                "0.1",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(os.path.join(h5_dir, "train.h5"))
        assert os.path.exists(os.path.join(h5_dir, "val.h5"))
        assert os.path.exists(os.path.join(h5_dir, "test.h5"))

        from pypots.data.saving.h5 import load_dict_from_h5

        # val set should have more missing than X_ori
        val_data = load_dict_from_h5(os.path.join(h5_dir, "val.h5"))
        assert "X" in val_data
        assert "X_ori" in val_data
        x_missing = np.isnan(val_data["X"]).sum()
        x_ori_missing = np.isnan(val_data["X_ori"]).sum()
        assert x_missing >= x_ori_missing, "val X should have more missing than X_ori"

    @pytest.mark.xdist_group(name="cli-data")
    def test_9_prepare_classification(self):
        """Test data prepare extracts classification labels."""
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "claf_data.csv")
        rows = []
        for sid in range(10):
            for step in range(5):
                rows.append(
                    {
                        "SAMPLE_ID": sid,
                        "feat_0": np.random.randn(),
                        "feat_1": np.random.randn(),
                        "CLAF_TARGET": sid % 3,
                    }
                )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        h5_path = os.path.join(self.temp_dir, "claf.h5")
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["prepare", "--input", csv_path, "--output", h5_path, "--task", "classification", "--set_type", "train"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        from pypots.data.saving.h5 import load_dict_from_h5

        loaded = load_dict_from_h5(h5_path)
        assert "y" in loaded
        assert loaded["y"].shape == (10,)
        assert len(np.unique(loaded["y"])) == 3

    @pytest.mark.xdist_group(name="cli-data")
    @pytest.mark.xfail(reason="Allow test to fail if ai4ts not installed")
    def test_10_profile(self):
        """Test data profile command."""
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "profile_test.csv")
        np.random.seed(42)
        rows = []
        for sid in range(5):
            for step in range(6):
                rows.append({"SAMPLE_ID": sid, "feat_0": np.random.randn(), "feat_1": np.random.randn()})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(data, ["profile", "--input", csv_path], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "Samples:" in result.output
        assert "Features:" in result.output
        assert "Strategy:" in result.output or "strategy" in result.output.lower()

    @pytest.mark.xdist_group(name="cli-data")
    @pytest.mark.xfail(reason="Allow test to fail if ai4ts not installed")
    def test_11_profile_json(self):
        """Test data profile with --json flag."""
        import json
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "profile_json.csv")
        np.random.seed(42)
        rows = []
        for sid in range(3):
            for step in range(4):
                rows.append({"SAMPLE_ID": sid, "feat_0": np.random.randn()})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(data, ["profile", "--input", csv_path, "--json"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        # Filter out ai4ts banner lines and parse JSON
        lines = result.output.strip().split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break
        assert json_start is not None, f"No JSON found in output: {result.output}"
        json_text = "\n".join(lines[json_start:])
        parsed = json.loads(json_text)
        assert parsed["dataset_stats"]["n_samples"] == 3
        assert parsed["dataset_stats"]["n_features"] == 1

    @pytest.mark.xdist_group(name="cli-data")
    @pytest.mark.xfail(reason="Allow test to fail if ai4ts not installed")
    def test_12_prepare_with_registry(self):
        """Test data prepare creates registry file when pipeline available."""
        import pandas as pd

        csv_path = os.path.join(self.temp_dir, "registry_test.csv")
        np.random.seed(42)
        rows = []
        for sid in range(8):
            for step in range(10):
                row = {"SAMPLE_ID": sid}
                for f in range(3):
                    row[f"feat_{f}"] = np.random.randn()
                rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        h5_path = os.path.join(self.temp_dir, "reg_train.h5")
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["prepare", "--input", csv_path, "--output", h5_path,
             "--task", "imputation", "--set_type", "train"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(h5_path)
        # A registry JSON should be created alongside
        expected_reg = os.path.splitext(h5_path)[0] + "_registry.json"
        assert os.path.exists(expected_reg), f"Registry not found at {expected_reg}"

    @pytest.mark.xdist_group(name="cli-data")
    @pytest.mark.xfail(reason="Allow test to fail if ai4ts not installed")
    def test_13_reconstruct(self):
        """Test data reconstruct command end-to-end."""
        import h5py
        import pandas as pd

        from pypots.data.saving.h5 import load_dict_from_h5

        # Create test CSV
        csv_path = os.path.join(self.temp_dir, "recon_input.csv")
        np.random.seed(42)
        rows = []
        for sid in range(5):
            for step in range(12):
                rows.append({"SAMPLE_ID": sid, "feat_0": np.random.randn(), "feat_1": np.random.randn()})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Prepare
        h5_path = os.path.join(self.temp_dir, "recon_train.h5")
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["prepare", "--input", csv_path, "--output", h5_path,
             "--task", "imputation", "--set_type", "train"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        reg_path = os.path.splitext(h5_path)[0] + "_registry.json"

        # Create "predictions" by copying X
        loaded = load_dict_from_h5(h5_path)
        pred_path = os.path.join(self.temp_dir, "predictions.h5")
        with h5py.File(pred_path, "w") as f:
            f.create_dataset("X", data=loaded["X"])

        # Reconstruct
        out_csv = os.path.join(self.temp_dir, "reconstructed.csv")
        result = runner.invoke(
            data,
            ["reconstruct", "--predictions", pred_path,
             "--registry", reg_path, "--output", out_csv],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(out_csv)
        recon_df = pd.read_csv(out_csv)
        assert "SAMPLE_ID" in recon_df.columns
        assert len(recon_df) == 60  # 5 samples × 12 steps


if __name__ == "__main__":
    unittest.main()
