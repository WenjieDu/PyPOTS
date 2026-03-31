"""
Test cases for the functions and classes in package `pypots.cli.model`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import unittest
from argparse import Namespace
from copy import copy

import pytest

from pypots.cli.model import model_command_factory
from tests.cli.config import PROJECT_ROOT_DIR


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIModel(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "action": None,
        "task": None,
        "name": None,
        "path": None,
        "output": None,
    }
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-model")
    def test_0_list(self):
        arguments = copy(self.default_arguments)
        arguments["action"] = "list"
        args = Namespace(**arguments)
        model_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-model")
    def test_1_list_with_task(self):
        arguments = copy(self.default_arguments)
        arguments["action"] = "list"
        arguments["task"] = "imputation"
        args = Namespace(**arguments)
        model_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-model")
    def test_2_describe(self):
        arguments = copy(self.default_arguments)
        arguments["action"] = "describe"
        arguments["name"] = "SAITS"
        arguments["task"] = "imputation"
        args = Namespace(**arguments)
        model_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-model")
    def test_3_config(self):
        arguments = copy(self.default_arguments)
        arguments["action"] = "config"
        arguments["name"] = "SAITS"
        arguments["task"] = "imputation"
        args = Namespace(**arguments)
        model_command_factory(args).run()


if __name__ == "__main__":
    unittest.main()
