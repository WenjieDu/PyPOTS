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
            "--lint_code",
            dest="lint_code",
            action="store_true",
            help="run Black and Flake8 to lint code",
        )
        train_parser.set_defaults(func=dev_command_factory)

    def __init__(
        self,
        run_tests: bool,
        lint_code: bool,
    ):
        self._run_tests = run_tests
        self._lint_code = lint_code

    def run(self):
        print(f"current dir: {os.getcwd()}")

        if self._run_tests:
            try:
                os.system("pytest")
                os.system("rm -rf .pytest_cache")
            except Exception as e:
                raise RuntimeError(e)
        elif self._lint_code:
            try:
                os.system("black .")
                os.system("flake8 .")
            except Exception as e:
                raise RuntimeError(e)
