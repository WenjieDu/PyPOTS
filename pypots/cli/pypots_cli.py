"""
PyPOTS CLI (Command Line Interface) tool, built with Click.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import click

from .benchmark import benchmark
from .data import data
from .dev import dev
from .doc import doc
from .env import env
from .evaluate import evaluate
from .info import info
from .model import model
from .predict import predict
from .train import train
from .tune import tune


@click.group(name="pypots-cli", help="PyPOTS Command-Line-Interface tool")
def cli():
    """PyPOTS CLI — a command-line tool for managing PyPOTS models, data, training, and more."""
    pass


cli.add_command(benchmark)
cli.add_command(data)
cli.add_command(dev)
cli.add_command(doc)
cli.add_command(env)
cli.add_command(evaluate)
cli.add_command(info)
cli.add_command(model)
cli.add_command(predict)
cli.add_command(train)
cli.add_command(tune)


def main():
    cli()


if __name__ == "__main__":
    main()
