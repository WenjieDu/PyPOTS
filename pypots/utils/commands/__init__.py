"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
