"""
Shared utilities for PyPOTS CLI commands.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
import subprocess
import sys


def execute_command(command: str, verbose: bool = True):
    """Run a shell command, streaming output to stdout/stderr when verbose.

    Parameters
    ----------
    command : str
        The shell command to execute.
    verbose : bool
        If True, stream output in real time. If False, capture it.

    Returns
    -------
    exec_result :
        The completed process result.
    """
    from ..utils.logging import logger

    logger.info(f"Executing '{command}'...")
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
        raise RuntimeError(exec_result.stdout, exec_result.stderr)
    return exec_result


def check_if_under_root_dir(strict: bool = True):
    """Check if under the root dir of PyPOTS project.

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
            "pyproject.toml",
        }
    )

    if strict:
        if not check_result:
            raise RuntimeError(
                "Command `pypots-cli dev` can only be run under the root directory of project PyPOTS, "
                f"but you're running it under the path {os.getcwd()}. Please make a check."
            )

    return check_result
