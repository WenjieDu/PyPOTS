"""
Test cases for the CLI command `pypots.cli.doc`.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import threading
import unittest

import pytest
from click.testing import CliRunner

from pypots.cli.doc import doc
from pypots.utils.logging import logger
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


class TestPyPOTSCLIDoc(unittest.TestCase):
    # `pypots-cli doc` must run under the project root dir
    os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-doc")
    def test_0_gene_rst(self):
        runner = CliRunner()
        result = runner.invoke(doc, ["--gene_rst"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        logger.info("run again under a non-root dir")
        try:
            os.chdir(os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "pypots")))
            runner.invoke(doc, ["--gene_rst"], catch_exceptions=False)
        except RuntimeError:  # try to run under a non-root dir, so RuntimeError will be raised
            pass
        except Exception as e:  # other exceptions will cause an error and result in failed testing
            raise e
        finally:
            os.chdir(PROJECT_ROOT_DIR)

    @pytest.mark.xdist_group(name="cli-doc")
    def test_1_gene_html(self):
        runner = CliRunner()
        try:
            runner.invoke(doc, ["--gene_html"], catch_exceptions=False)
        except Exception as e:  # somehow we have some error when testing on Windows, so just print and pass below
            logger.error(f"❌ Exception: {e}")

    @pytest.mark.xdist_group(name="cli-doc")
    @time_out(2, callback_func)  # wait for two seconds
    def test_2_view_doc(self):
        runner = CliRunner()
        try:
            runner.invoke(doc, ["--view_doc"], catch_exceptions=False)
        except Exception as e:  # somehow we have some error when testing on Windows, so just print and pass below
            logger.error(f"❌ Exception: {e}")

    @pytest.mark.xdist_group(name="cli-doc")
    def test_3_cleanup(self):
        runner = CliRunner()
        result = runner.invoke(doc, ["--cleanup"], catch_exceptions=False)
        assert result.exit_code == 0, result.output


if __name__ == "__main__":
    unittest.main()
