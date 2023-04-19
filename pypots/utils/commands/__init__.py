"""
Expose the base class for PyPOTS CLI (command line interface).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


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

    @abstractmethod
    def run(self):
        raise NotImplementedError()
