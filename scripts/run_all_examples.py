"""Run public examples as independent subprocesses and propagate pytest failures."""

# License: BSD-3-Clause

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SMOKE_EXAMPLES = (
    "imputation/saits_imputation_example.py",
    "forecasting/dlinear_forecasting_example.py",
    "classification/saits_classification_example.py",
    "clustering/crli_clustering_example.py",
    "anomaly_detection/saits_anomalydetection_example.py",
    "representation/ts2vec_representation_example.py",
)
# Isolate examples whose models may fetch pretrained weights from external services.
NETWORK_MODELS = {"gpt4ts", "moment", "timellm", "tsicl"}


def select_examples(scope):
    scripts = sorted(
        path for path in EXAMPLES.rglob("*.py") if not path.name.startswith("__") and "checkpoint" not in str(path)
    )
    if scope == "smoke":
        selected = [EXAMPLES / name for name in SMOKE_EXAMPLES]
    elif scope == "all":
        selected = scripts
    else:
        selected = [path for path in scripts if (path.name.split("_", 1)[0] in NETWORK_MODELS) == (scope == "network")]
    if not selected:
        raise ValueError(f"No examples selected for {scope}")
    return selected


@pytest.mark.parametrize(
    "script_path",
    select_examples(os.environ.get("PYPOTS_EXAMPLE_SCOPE", "all")),
    ids=lambda path: path.relative_to(EXAMPLES).as_posix(),
)
def test_standalone_examples_can_run(script_path):
    """Execute the public script with the same interpreter as pytest."""
    assert script_path.is_file(), f"Missing example: {script_path}"
    env = os.environ.copy()
    if env.get("PYPOTS_EXAMPLE_SCOPE") in {"offline", "smoke"}:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script_path.name} exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "offline", "network", "all"), default="all")
    parser.add_argument("--workers", default="1")
    parser.add_argument("--junitxml")
    args = parser.parse_args()
    os.environ["PYPOTS_EXAMPLE_SCOPE"] = args.scope
    pytest_args = ["-rA", "-n", args.workers, __file__]
    if args.junitxml:
        pytest_args.extend(["--junitxml", args.junitxml])
    raise SystemExit(pytest.main(pytest_args))
