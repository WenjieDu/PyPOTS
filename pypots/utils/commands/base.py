"""
The base class for PyPOTS CLI (command line interface).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import os
import subprocess
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from pypots.utils.logging import logger


class BaseCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @staticmethod
    def execute_command(command: str, verbose: bool = True):
        if verbose:
            exec_result = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                close_fds=True,
                stderr=sys.stderr,
                stdout=sys.stdout,
                universal_newlines=True,
                shell=True,
                bufsize=1,
            )
            exec_result.communicate()
        else:
            exec_result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            if exec_result.returncode != 0:
                if len(exec_result.stderr) > 0:
                    logger.error(exec_result.stderr)
                if len(exec_result.stdout) > 0:
                    logger.error(exec_result.stdout)
                raise RuntimeError()
        return exec_result.returncode

    @staticmethod
    def check_if_under_root_dir(strict: bool = True):
        """ Check if under the root dir of PyPOTS project.

        Parameters
        ----------
        strict : bool, default = True,
            Whether to raise a RuntimeError if currently not under the root dir of PyPOTS project.

        Returns
        -------
        check_result : bool,
            Whether currently under the root dir of PyPOTS project.
        """
        all_files_under_current_dir = set(os.listdir("."))
        check_result = all_files_under_current_dir.issuperset(
            {
                ".github",
                "docs",
                "pypots",
                "tutorials",
                "setup.cfg",
                "setup.py",
            }
        )

        if strict:
            assert check_result, RuntimeError(
                "Command `pypots-cli dev` can only be run under the root directory of project PyPOTS, "
                f"but you're running it under the path {os.getcwd()}. Please make a check."
            )

        return check_result

    @abstractmethod
    def run(self):
        raise NotImplementedError()
