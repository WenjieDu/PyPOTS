"""
This test is used for unified and automated coverage of all independent Python scripts in the `examples/` directory.
It ensures that the demo scripts provided for beginners run perfectly during every CI run,
preventing subsequent PyPOTS updates from causing the example code to fail.
"""

import os
import glob
import subprocess
import pytest

from pypots.utils.logging import logger

# 1. Scan all files ending with `.py` in the examples folder
# and exclude system or hidden files that are not genuine example scripts.
examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
example_scripts = glob.glob(os.path.join(examples_dir, "**/*.py"), recursive=True)

# Filter out valid scripts for test execution (e.g., exclude __init__.py or temporary files)
valid_scripts = [
    script for script in example_scripts if not os.path.basename(script).startswith("__") and "checkpoint" not in script
]


@pytest.mark.parametrize("script_path", valid_scripts)
def test_standalone_examples_can_run(script_path):
    """
    We execute each collected example script as an independent subprocess.
    If the subprocess exits normally (exit code == 0), the example code is valid and correct.
    Otherwise, the unit test will immediately fail, alerting the developer to fix the corresponding Example.
    """
    script_name = os.path.relpath(script_path, examples_dir)
    logger.info(f"🚀 [Testing Example Script] Running as subprocess: {script_name}...")

    # Setup env to use local pypots package instead of the installed one
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root

    # Execute the command. A longer timeout is set here to ensure even larger examples can finish training.
    # We expect every example script to run properly as if a beginner executes `python example.py`.
    result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=180, env=env)

    # Verify if the subprocess exited successfully
    if result.returncode != 0:
        logger.error(
            f"❌ '{script_name}' execution failed!\n\nStandard Output:\n{result.stdout}\n\nStandard Error:\n{result.stderr}"
        )
        pytest.fail(
            f"Example code {script_name} execution failed. Please check if it's incompatible with the latest API or parameters of the framework."
        )

    logger.info(f"✅ '{script_name}' successfully executed!")


if __name__ == "__main__":
    pytest.main(["-s", "-n", "auto", __file__])
