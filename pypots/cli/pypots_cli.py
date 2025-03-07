"""
PyPOTS CLI (Command Line Interface) tool
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from argparse import ArgumentParser

from .dev import DevCommand
from .doc import DocCommand
from .env import EnvCommand
from .hpo import HPOCommand


def main():
    parser = ArgumentParser("PyPOTS Command-Line-Interface tool", usage="pypots-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="pypots-cli command helpers")

    # Register commands here
    DevCommand.register_subcommand(commands_parser)
    DocCommand.register_subcommand(commands_parser)
    EnvCommand.register_subcommand(commands_parser)
    HPOCommand.register_subcommand(commands_parser)

    # parse all arguments
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # then run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
