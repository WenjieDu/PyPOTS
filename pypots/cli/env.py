"""
CLI tools to help initialize environments for running and developing PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

try:
    import torch

    # import numpy
    # import sklearn
    # import pandas
    # import tensorboard
    # import scipy
    # import h5py
    # import tsdb
    # import pygrinder
except ImportError:
    raise ImportError(
        "Torch not installed. Using this tool supposes that you've already installed `pypots` "
        "with at least basic dependencies in requirements/requirements.txt in PyPOTS project's root dir."
    )

from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from ..utils.logging import logger


def env_command_factory(args: Namespace):
    return EnvCommand(
        args.install,
        args.tool,
    )


class EnvCommand(BaseCommand):
    """CLI tools helping users and developer setup python environments for running and developing PyPOTS.

    Notes
    -----
    Using this tool supposes that you've already installed `pypots` with at least basic dependencies.
    Please refer to file requirements/requirements.txt in PyPOTS project's root dir for references.

    Examples
    --------
    $ pypots-cli env --scope full --tool pip
    $ pypots-cli env --scope full --tool pip
    $ pypots-cli env --scope dev --tool conda -n
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "env",
            help="CLI tools helping users and developer setup python environments for running and developing PyPOTS",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "--install",
            dest="install",
            type=str,
            required=True,
            choices=["dev", "full", "doc", "test", "optional"],
            help="Install specified dependencies in the current python environment",
        )
        sub_parser.add_argument(
            "--tool",
            dest="tool",
            type=str,
            required=True,
            choices=["conda", "pip"],
            help="Setup the environment with pip or conda, have to be specific",
        )

        sub_parser.set_defaults(func=env_command_factory)

    def __init__(
        self,
        install: bool,
        tool: str,
    ):
        self._install = install
        self._tool = tool

    def checkup(self):
        """Run some checks on the arguments to avoid error usages"""
        self.check_if_under_root_dir(strict=True)

    def run(self):
        """Execute the given command."""
        # run checks first
        self.checkup()

        logger.info(f"Installing the dependencies in scope `{self._install}` for you...")

        if self._tool == "conda":
            assert (
                self.execute_command("which conda").returncode == 0
            ), "Conda not installed, cannot set --tool=conda, please check your conda."

            self.execute_command("conda install pyg pytorch-scatter pytorch-sparse -c pyg")

        else:  # self._tool == "pip"
            torch_version = torch.__version__

            if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
                if "cpu" not in torch_version:
                    torch_version = torch_version + "+cpu"

            self.execute_command(
                f"python -m pip install -e '.[optional]' -f https://data.pyg.org/whl/torch-{torch_version}.html"
            )

            if self._install != "optional":
                self.execute_command(f"pip install -e '.[{self._install}]'")
        logger.info("Installation finished. Enjoy your play with PyPOTS! Bye ;-)")
