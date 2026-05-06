"""
Test cases for the CLI command `pypots.cli.model`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest
from click.testing import CliRunner

from pypots.cli.model import model
from tests.cli.config import PROJECT_ROOT_DIR


class TestPyPOTSCLIModel(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-model")
    def test_0_list(self):
        runner = CliRunner()
        result = runner.invoke(model, ["list"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-model")
    def test_1_list_with_task(self):
        runner = CliRunner()
        result = runner.invoke(model, ["list", "--task", "imputation"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-model")
    def test_2_describe(self):
        runner = CliRunner()
        result = runner.invoke(
            model,
            ["describe", "--name", "SAITS", "--task", "imputation"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-model")
    def test_3_config(self):
        runner = CliRunner()
        result = runner.invoke(
            model,
            ["config", "--name", "SAITS", "--task", "imputation"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
