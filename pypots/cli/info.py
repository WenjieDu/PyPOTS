"""
CLI command to display PyPOTS environment information.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import platform

import click

from .base import safe_print
from .utils import SUPPORTED_TASKS, list_available_models


@click.command(name="info", help="Display PyPOTS version, environment, and device information")
def info():
    """Execute the info command."""
    from ..version import __version__

    safe_print("=" * 60)
    safe_print("PyPOTS Environment Information")
    safe_print("=" * 60)

    # PyPOTS version
    safe_print(f"\n{'PyPOTS version:':<30} {__version__}")

    # Python version
    safe_print(f"{'Python version:':<30} {platform.python_version()}")

    # OS info
    safe_print(f"{'Operating system:':<30} {platform.system()} {platform.release()}")
    safe_print(f"{'Platform:':<30} {platform.platform()}")

    # PyTorch info
    try:
        import torch

        safe_print(f"\n{'PyTorch version:':<30} {torch.__version__}")
        safe_print(f"{'CUDA available:':<30} {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            safe_print(f"{'CUDA version:':<30} {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            safe_print(f"{'GPU count:':<30} {gpu_count}")
            for i in range(gpu_count):
                safe_print(f"{'  GPU ' + str(i) + ':':<30} {torch.cuda.get_device_name(i)}")

        # MPS (Apple Silicon) support
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        safe_print(f"{'MPS available:':<30} {has_mps}")

        # Ascend NPU support
        try:
            import torch_npu

            npu_available = torch_npu.npu.is_available()
            safe_print(f"{'NPU available:':<30} {npu_available}")
            if npu_available:
                safe_print(f"{'torch_npu version:':<30} {torch_npu.__version__}")
                npu_count = torch_npu.npu.device_count()
                safe_print(f"{'NPU count:':<30} {npu_count}")
                for i in range(npu_count):
                    safe_print(f"{'  NPU ' + str(i) + ':':<30} {torch_npu.npu.get_device_name(i)}")
        except ImportError:
            safe_print(f"{'NPU available:':<30} False (torch_npu not installed)")
    except ImportError:
        safe_print("\nPyTorch:                       NOT INSTALLED")

    # Available models per task
    safe_print("\n" + "-" * 60)
    safe_print("Available Models")
    safe_print("-" * 60)
    try:
        models = list_available_models()
        total = 0
        for task in SUPPORTED_TASKS:
            count = len(models.get(task, []))
            total += count
            safe_print(f"  {task:<25} {count} models")
        safe_print(f"  {'TOTAL':<25} {total} models")
    except Exception as e:  # noqa: BLE001 - keep optional model inventory failures local to CLI info
        safe_print(f"  Error listing models: {e}")

    # Optional dependencies status
    safe_print("\n" + "-" * 60)
    safe_print("Optional Dependencies")
    safe_print("-" * 60)
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
        safe_print(f"  {description:<45} {status}")

    safe_print("\n" + "=" * 60)
