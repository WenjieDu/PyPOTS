"""
CLI command to display PyPOTS environment information.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import platform

import click

from .utils import SUPPORTED_TASKS, list_available_models


@click.command(name="info", help="Display PyPOTS version, environment, and device information")
def info():
    """Execute the info command."""
    from ..version import __version__

    print("=" * 60)
    print("PyPOTS Environment Information")
    print("=" * 60)

    # PyPOTS version
    print(f"\n{'PyPOTS version:':<30} {__version__}")

    # Python version
    print(f"{'Python version:':<30} {platform.python_version()}")

    # OS info
    print(f"{'Operating system:':<30} {platform.system()} {platform.release()}")
    print(f"{'Platform:':<30} {platform.platform()}")

    # PyTorch info
    try:
        import torch

        print(f"\n{'PyTorch version:':<30} {torch.__version__}")
        print(f"{'CUDA available:':<30} {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"{'CUDA version:':<30} {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"{'GPU count:':<30} {gpu_count}")
            for i in range(gpu_count):
                print(f"{'  GPU ' + str(i) + ':':<30} {torch.cuda.get_device_name(i)}")

        # MPS (Apple Silicon) support
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"{'MPS available:':<30} {has_mps}")
    except ImportError:
        print("\nPyTorch:                       NOT INSTALLED")

    # Available models per task
    print("\n" + "-" * 60)
    print("Available Models")
    print("-" * 60)
    try:
        models = list_available_models()
        total = 0
        for task in SUPPORTED_TASKS:
            count = len(models.get(task, []))
            total += count
            print(f"  {task:<25} {count} models")
        print(f"  {'TOTAL':<25} {total} models")
    except Exception as e:
        print(f"  Error listing models: {e}")

    # Optional dependencies status
    print("\n" + "-" * 60)
    print("Optional Dependencies")
    print("-" * 60)
    optional_deps = {
        "tensorboard": "TensorBoard (training visualization)",
        "optuna": "Optuna (hyperparameter optimization)",
        "yaml": "PyYAML (YAML config files)",
        "scipy": "SciPy (scientific computing)",
        "h5py": "h5py (HDF5 data files)",
        "pandas": "Pandas (data manipulation)",
        "matplotlib": "Matplotlib (plotting)",
        "pygrinder": "PyGrinder (missing data simulation)",
        "tsdb": "TSDB (time-series database)",
        "benchpots": "BenchPOTS (benchmarking)",
    }
    for pkg, description in optional_deps.items():
        try:
            __import__(pkg)
            status = "installed"
        except ImportError:
            status = "NOT installed"
        print(f"  {description:<45} {status}")

    print("\n" + "=" * 60)
