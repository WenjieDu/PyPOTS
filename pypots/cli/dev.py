"""
CLI tools to help the development team build PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
import shutil

import click

from .base import execute_command, check_if_under_root_dir

IMPORT_ERROR_MESSAGE = (
    "`pypots-cli dev` command is for PyPOTS developers to run tests easily. "
    "Therefore, you need a complete PyPOTS development environment. However, you are missing some dependencies. "
    "Please refer to https://github.com/WenjieDu/PyPOTS/blob/main/environment-dev.yml for dependency details. "
)


@click.command(name="dev", help="CLI tools helping develop PyPOTS code")
@click.option(
    "--build",
    is_flag=True,
    help="Build PyPOTS into a wheel and package the source code into a .tar.gz file for distribution",
)
@click.option(
    "-c",
    "--cleanup",
    is_flag=True,
    help="Delete all caches and building files",
)
@click.option(
    "--run_tests",
    "--run-tests",
    is_flag=True,
    help="Run all test cases",
)
@click.option(
    "-k",
    "k",
    default=None,
    type=str,
    help="The -k option of pytest. Description of -k option in pytest: "
    "only run tests which match the given substring expression. An expression is a python evaluatable "
    "expression where all names are substring-matched against test names and their parent classes. "
    "Example: -k 'test_method or test_other' matches all test functions and classes whose name contains "
    "'test_method' or 'test_other', while -k 'not test_method' matches those that don't contain "
    "'test_method' in their names. -k 'not test_method and not test_other' will eliminate the matches. "
    "Additionally keywords are matched to classes and functions containing extra names in their "
    "'extra_keyword_matches' set, as well as functions which have names assigned directly to them. The "
    "matching is case-insensitive.",
)
@click.option(
    "--show-coverage",
    "--show_coverage",
    is_flag=True,
    help="Show the code coverage report after running tests",
)
@click.option(
    "--lint-code",
    "--lint_code",
    is_flag=True,
    help="Run Black and Flake8 to lint code",
)
def dev(build, cleanup, run_tests, k, show_coverage, lint_code):
    """Execute the dev command."""
    from ..utils.logging import logger

    # run checks
    check_if_under_root_dir(strict=True)

    if k is not None:
        assert run_tests, (
            "Argument `-k` should combine the use of `--run_tests`. Try `pypots-cli dev --run_tests -k your_pattern`"
        )

    if show_coverage:
        assert run_tests, (
            "Argument `--show_coverage` should combine the use of `--run_tests`. "
            "Try `pypots-cli dev --run_tests --show_coverage`"
        )

    if cleanup:
        assert not run_tests and not lint_code, (
            "Argument `--cleanup` should be used alone. Try `pypots-cli dev --cleanup`"
        )

    try:
        if cleanup:
            shutil.rmtree("build", ignore_errors=True)
            shutil.rmtree("dist", ignore_errors=True)
            shutil.rmtree("pypots.egg-info", ignore_errors=True)
        elif build:
            execute_command("python -m build")
        elif run_tests:
            pytest_command = f"pytest -k {k}" if k is not None else "pytest"
            command_to_run_test = f"coverage run -m {pytest_command}" if show_coverage else pytest_command
            execute_command(command_to_run_test)
            if show_coverage and os.path.exists(".coverage"):
                execute_command("coverage report -m")
        elif lint_code:
            logger.info("Reformatting with Ruff...")
            execute_command("ruff format .")
            logger.info("Linting with Ruff...")
            execute_command("ruff check .")
    except ImportError:
        raise ImportError(IMPORT_ERROR_MESSAGE)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(".pytest_cache", ignore_errors=True)
