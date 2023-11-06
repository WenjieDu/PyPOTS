"""
Test cases for the functions and classes in package `pypots.cli.dev`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import threading
import unittest
from argparse import Namespace
from copy import copy

import pytest

from pypots.cli.dev import dev_command_factory
from tests.cli.config import PROJECT_ROOT_DIR


def callback_func():
    raise TimeoutError("Time out.")


def time_out(interval, callback):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t = threading.Thread(target=func, args=args, kwargs=kwargs)
            t.setDaemon(True)
            t.start()
            t.join(interval)  # wait for interval seconds
            if t.is_alive():
                return threading.Timer(0, callback).start()  # invoke callback()
            else:
                return

        return wrapper

    return decorator


@pytest.mark.xfail(reason="Allow tests for CLI to fail")
class TestPyPOTSCLIDev(unittest.TestCase):
    # set up the default arguments
    default_arguments = {
        "build": False,
        "cleanup": False,
        "run_tests": False,
        "k": None,
        "show_coverage": False,
        "lint_code": False,
    }
    # `pypots-cli dev` must run under the project root dir
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-dev")
    def test_0_build(self):
        arguments = copy(self.default_arguments)
        arguments["build"] = True
        args = Namespace(**arguments)
        dev_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-dev")
    def test_1_run_tests(self):
        arguments = copy(self.default_arguments)
        arguments["run_tests"] = True
        arguments["k"] = "try_to_find_a_non_existing_test_case"
        args = Namespace(**arguments)
        try:
            dev_command_factory(args).run()
        except RuntimeError:  # try to find a non-existing test case, so RuntimeError will be raised
            pass
        except Exception as e:  # other exceptions will cause an error and result in failed testing
            raise e

    # Don't test --lint-code because Black will reformat the code and cause error when generating the coverage report
    # @pytest.mark.xdist_group(name="cli-dev")
    # def test_2_lint_code(self):
    #     arguments = copy(self.default_arguments)
    #     arguments["lint_code"] = True
    #     args = Namespace(**arguments)
    #     dev_command_factory(args).run()

    @pytest.mark.xdist_group(name="cli-dev")
    def test_3_cleanup(self):
        arguments = copy(self.default_arguments)
        arguments["cleanup"] = True
        args = Namespace(**arguments)
        dev_command_factory(args).run()


if __name__ == "__main__":
    unittest.main()
