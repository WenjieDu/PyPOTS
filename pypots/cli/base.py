"""
Shared utilities for PyPOTS CLI commands.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import io
import os
import subprocess
import sys


def _has_real_fileno(stream) -> bool:
    """Return True if *stream* backs a real OS file descriptor.

    Click's CliRunner (used in tests) replaces sys.stdout / sys.stderr with
    in-memory StringIO objects that raise ``io.UnsupportedOperation`` when
    ``.fileno()`` is called.  subprocess.Popen requires a real fd when a
    stream is passed directly, so we must detect this situation and fall back
    to subprocess.PIPE + manual forwarding instead.
    """
    try:
        stream.fileno()
        return True
    except (AttributeError, io.UnsupportedOperation):
        return False


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
        # When sys.stdout / sys.stderr are real file objects (normal CLI usage)
        # pass them directly to Popen so output is streamed in real time.
        # When they are in-memory wrappers (e.g. Click's CliRunner in tests)
        # they have no underlying fd, so we capture via PIPE and forward
        # manually – this also lets the test runner capture the output.
        stdout_dest = sys.stdout if _has_real_fileno(sys.stdout) else subprocess.PIPE
        stderr_dest = sys.stderr if _has_real_fileno(sys.stderr) else subprocess.PIPE

        exec_result = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            close_fds=True,
            stdout=stdout_dest,
            stderr=stderr_dest,
            universal_newlines=True,
            shell=True,
            bufsize=1,
        )
        stdout_output, stderr_output = exec_result.communicate()

        # Forward captured output to sys.stdout/sys.stderr (no-op when the
        # streams were passed directly, because communicate() returns None).
        if stdout_output:
            sys.stdout.write(stdout_output)
        if stderr_output:
            sys.stderr.write(stderr_output)
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
