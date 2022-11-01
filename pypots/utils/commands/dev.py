"""
CLI tools to help the development team build PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
from argparse import ArgumentParser, Namespace

from pypots.utils.commands import BaseCommand


def dev_command_factory(args: Namespace):
    return DevCommand(
        args.run_tests,
        args.k,
        args.show_coverage,
        args.lint_code,
    )


class DevCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        train_parser = parser.add_parser("dev", help="CLI tool to help development ")
        train_parser.add_argument(
            "--run_tests",
            dest="run_tests",
            action="store_true",
            help="run all test cases",
        )
        train_parser.add_argument(
            "--show_coverage",
            dest="show_coverage",
            action="store_true",
            help="show the code coverage report after running tests",
        )
        train_parser.add_argument(
            "-k",
            type=str,
            default="",
            required=False,
            help="the -k option of pytest. Description of -k option in pytest: "
            "only run tests which match the given substring expression. An expression is a python evaluatable "
            "expression where all names are substring-matched against test names and their parent classes. "
            "Example: -k 'test_method or test_other' matches all test functions and classes whose name contains "
            "'test_method' or 'test_other', while -k 'not test_method' matches those that don't contain "
            "'test_method' in their names. -k 'not test_method and not test_other' will eliminate the matches. "
            "Additionally keywords are matched to classes and functions containing extra names in their "
            "'extra_keyword_matches' set, as well as functions which have names assigned directly to them. The "
            "matching is case-insensitive.",
        )
        train_parser.add_argument(
            "--lint_code",
            dest="lint_code",
            action="store_true",
            help="run Black and Flake8 to lint code",
        )
        train_parser.set_defaults(func=dev_command_factory)

    def __init__(
        self,
        run_tests: bool,
        k: str,
        show_coverage: bool,
        lint_code: bool,
    ):
        self._run_tests = run_tests
        self._k = k
        self._show_coverage = show_coverage
        self._lint_code = lint_code

    def run(self):
        print(f"current dir: {os.getcwd()}")

        if self._run_tests:
            try:
                pytest_command = f"pytest -k {self._k}" if self._k else "pytest"
                command_to_run_test = (
                    f"coverage run -m {pytest_command}"
                    if self._show_coverage
                    else pytest_command
                )

                os.system(command_to_run_test)
                if self._show_coverage:
                    os.system("coverage report -m")
                    os.system("rm -rf .coverage")
                else:
                    print(
                        "Omit the code coverage report. Enable it by using --show_coverage if in need."
                    )
                os.system("rm -rf .pytest_cache")

            except Exception as e:
                raise RuntimeError(e)
        elif self._lint_code:
            try:
                os.system("black .")
                os.system("flake8 .")
            except Exception as e:
                raise RuntimeError(e)
