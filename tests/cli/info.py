"""
Test cases for the CLI command `pypots.cli.info`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest

import pytest
from click.testing import CliRunner

from pypots.cli.info import info
from tests.cli.config import PROJECT_ROOT_DIR


class TestPyPOTSCLIInfo(unittest.TestCase):
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-info")
    def test_0_info(self):
        runner = CliRunner()
        result = runner.invoke(info, [], catch_exceptions=False)
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
