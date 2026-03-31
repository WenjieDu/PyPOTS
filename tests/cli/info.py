"""
Test cases for the functions and classes in package `pypots.cli.info`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest
from argparse import Namespace
from copy import copy

import pytest

from pypots.cli.info import info_command_factory
from tests.cli.config import PROJECT_ROOT_DIR


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIInfo(unittest.TestCase):
    # set up the default arguments (info command has no arguments)
    default_arguments = {}
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-info")
    def test_0_info(self):
        arguments = copy(self.default_arguments)
        args = Namespace(**arguments)
        info_command_factory(args).run()


if __name__ == "__main__":
    unittest.main()
