"""
CLI tools to help initialize environments for running and developing PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import click

from .base import execute_command, check_if_under_root_dir


@click.command(
    name="env",
    help="CLI tools helping users and developer setup python environments for running and developing PyPOTS",
)
@click.option(
    "--install",
    required=True,
    type=click.Choice(["dev", "full", "doc", "test", "optional"]),
    help="Install specified dependencies in the current python environment",
)
@click.option(
    "--tool",
    required=True,
    type=click.Choice(["conda", "pip"]),
    help="Setup the environment with pip or conda, have to be specific",
)
def env(install, tool):
    """Execute the env command."""
    import torch

    from ..utils.logging import logger

    # run checks
    check_if_under_root_dir(strict=True)

    logger.info(f"Installing the dependencies in scope `{install}` for you...")

    if tool == "conda":
        assert execute_command("which conda").returncode == 0, (
            "Conda not installed, cannot set --tool=conda, please check your conda."
        )

        execute_command("conda install pyg pytorch-scatter pytorch-sparse -c pyg")

    else:  # tool == "pip"
        torch_version = torch.__version__

        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            if "cpu" not in torch_version:
                torch_version = torch_version + "+cpu"

        execute_command(
            f"python -m pip install -e '.[optional]' -f https://data.pyg.org/whl/torch-{torch_version}.html"
        )

        if install != "optional":
            execute_command(f"pip install -e '.[{install}]'")
    logger.info("Installation finished. Enjoy your play with PyPOTS! Bye ;-)")
