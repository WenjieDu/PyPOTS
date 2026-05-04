"""
PyPOTS CLI (Command Line Interface) tool, built with Click.

Commands are lazy-loaded so that ``pypots-cli --help`` stays fast even when
heavy dependencies (torch, numpy, transformers …) are installed.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import importlib

import click


# Mapping from command name to the module attribute that holds the Click
# command/group object.  Format: "cmd_name": ("module_path", "attr_name")
_COMMAND_MAP = {
    "benchmark": ("pypots.cli.benchmark", "benchmark"),
    "data": ("pypots.cli.data", "data"),
    "dev": ("pypots.cli.dev", "dev"),
    "doc": ("pypots.cli.doc", "doc"),
    "env": ("pypots.cli.env", "env"),
    "evaluate": ("pypots.cli.evaluate", "evaluate"),
    "info": ("pypots.cli.info", "info"),
    "model": ("pypots.cli.model", "model"),
    "predict": ("pypots.cli.predict", "predict"),
    "recommend": ("pypots.cli.recommend", "recommend"),
    "train": ("pypots.cli.train", "train"),
    "tune": ("pypots.cli.tune", "tune"),
}


class LazyGroup(click.Group):
    """A Click Group that lazily imports command modules on first use.

    This avoids importing heavyweight libraries (torch, numpy, …) just to
    display ``--help`` for the top-level CLI.
    """

    def list_commands(self, ctx):
        return sorted(_COMMAND_MAP.keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name not in _COMMAND_MAP:
            return None
        module_path, attr_name = _COMMAND_MAP[cmd_name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name)


@click.group(cls=LazyGroup, name="pypots-cli",
             help="PyPOTS Command-Line-Interface tool")
def cli():
    """PyPOTS CLI — a command-line tool for managing PyPOTS models, data, training, and more."""
    pass


def main():
    cli()


if __name__ == "__main__":
    main()
