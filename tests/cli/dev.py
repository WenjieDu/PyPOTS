"""
Test cases for the CLI command `pypots.cli.dev`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import threading
import unittest

import pytest
from click.testing import CliRunner

from pypots.cli.dev import dev
from tests.cli.config import PROJECT_ROOT_DIR


def callback_func():
    raise TimeoutError("Time out.")


def time_out(interval, callback):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t = threading.Thread(target=func, args=args, kwargs=kwargs)
            t.daemon = True
            t.start()
            t.join(interval)  # wait for interval seconds
            if t.is_alive():
                return threading.Timer(0, callback).start()  # invoke callback()
            else:
                return

        return wrapper

    return decorator


class TestPyPOTSCLIDev(unittest.TestCase):
    # `pypots-cli dev` must run under the project root dir
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-dev")
    def test_0_build(self):
        runner = CliRunner()
        result = runner.invoke(dev, ["--build"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    @pytest.mark.xdist_group(name="cli-dev")
    def test_1_run_tests(self):
        runner = CliRunner()
        try:
            result = runner.invoke(
                dev,
                ["--run_tests", "-k", "try_to_find_a_non_existing_test_case"],
                catch_exceptions=False,
            )
            print(result.output)
        except RuntimeError:  # try to find a non-existing test case, so RuntimeError will be raised
            pass
        except Exception as e:  # other exceptions will cause an error and result in failed testing
            raise e

    # Don't test --lint-code because Black will reformat the code and cause error when generating the coverage report
    # @pytest.mark.xdist_group(name="cli-dev")
    # def test_2_lint_code(self):
    #     runner = CliRunner()
    #     result = runner.invoke(dev, ["--lint_code"], catch_exceptions=False)

    @pytest.mark.xdist_group(name="cli-dev")
    def test_3_cleanup(self):
        runner = CliRunner()
        result = runner.invoke(dev, ["--cleanup"], catch_exceptions=False)
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
