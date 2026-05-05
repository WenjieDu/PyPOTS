"""
PyPOTS package.

Submodules are lazy-loaded on first access to keep ``import pypots`` fast.
This allows CLI tools and lightweight scripts to start without pulling in
heavy dependencies (torch, transformers, etc.) that live deeper in the tree.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .version import __version__

__all__ = [
    "TimeSeriesAI",
    "imputation",
    "classification",
    "clustering",
    "forecasting",
    "anomaly_detection",
    "representation",
    "optim",
    "data",
    "utils",
    "__version__",
]

# Lazy-loaded submodules — imported on first attribute access
_LAZY_SUBMODULES = {
    "imputation",
    "classification",
    "clustering",
    "forecasting",
    "anomaly_detection",
    "representation",
    "optim",
    "data",
    "utils",
}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name == "TimeSeriesAI":
        from .timeseries_ai import TimeSeriesAI

        globals()["TimeSeriesAI"] = TimeSeriesAI
        return TimeSeriesAI

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
