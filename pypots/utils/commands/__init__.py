"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import subprocess
from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @staticmethod
    def execute_command(command):
        exec_result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        if exec_result.returncode != 0:
            if len(exec_result.stderr) > 0:
                raise RuntimeError(exec_result.stderr)
            if len(exec_result.stdout) > 0:
                raise RuntimeError(exec_result.stdout)

    @abstractmethod
    def run(self):
        raise NotImplementedError()
