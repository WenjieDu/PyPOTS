"""
Adding CLI utilities here, including config loading and the model registry.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect
import json
import os
import sys
from importlib import import_module, util
from types import ModuleType
from typing import Optional


def _get_logger():
    """Lazy-load the PyPOTS logger to avoid triggering heavy imports at module level."""
    from ..utils.logging import logger

    return logger


def load_package_from_path(pkg_path: str) -> ModuleType:
    """Load a package from a given path. Please refer to https://stackoverflow.com/a/50395128"""
    init_path = os.path.join(pkg_path, "__init__.py")
    assert os.path.exists(init_path)

    name = os.path.basename(pkg_path)
    spec = util.spec_from_file_location(name, init_path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Supported tasks and their corresponding module paths
# ---------------------------------------------------------------------------

SUPPORTED_TASKS = [
    "imputation",
    "classification",
    "forecasting",
    "anomaly_detection",
    "clustering",
    "representation",
]

TASK_MODULES = {
    "imputation": "pypots.imputation",
    "classification": "pypots.classification",
    "forecasting": "pypots.forecasting",
    "anomaly_detection": "pypots.anomaly_detection",
    "clustering": "pypots.clustering",
    "representation": "pypots.representation",
}

# Optimizer name -> module path mapping
OPTIMIZER_REGISTRY = {
    "Adam": "pypots.optim.Adam",
    "AdamW": "pypots.optim.AdamW",
    "Adagrad": "pypots.optim.Adagrad",
    "Adadelta": "pypots.optim.Adadelta",
    "RMSprop": "pypots.optim.RMSprop",
    "SGD": "pypots.optim.SGD",
}


def load_config(path: str) -> dict:
    """Load a YAML or JSON configuration file.

    Parameters
    ----------
    path : str
        Path to the configuration file. Format is auto-detected by extension.

    Returns
    -------
    config : dict
        The parsed configuration dictionary.
    """
    assert os.path.exists(path), f"Configuration file not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files. Install it with: pip install pyyaml")
            config = yaml.safe_load(f)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format '{ext}'. Use .yaml, .yml, or .json.")
    return config


def merge_config_with_overrides(config: dict, overrides: dict) -> dict:
    """Merge CLI argument overrides into a config dictionary.
    CLI arguments take precedence over config file values. Only non-None values are merged.

    Parameters
    ----------
    config : dict
        Base configuration dictionary loaded from a file.
    overrides : dict
        Dictionary of CLI argument overrides.

    Returns
    -------
    config : dict
        The merged configuration dictionary.
    """
    for key, val in overrides.items():
        if val is not None:
            config[key] = val
    return config


def get_model_class(task: str, model_name: str):
    """Lazily resolve a model class from a task name and model name.

    Parameters
    ----------
    task : str
        The task type, e.g. "imputation", "classification".
    model_name : str
        The model class name, e.g. "SAITS", "BRITS".

    Returns
    -------
    model_class : type
        The resolved model class.
    """
    assert task in SUPPORTED_TASKS, f"Unknown task '{task}'. Supported tasks: {SUPPORTED_TASKS}"
    module_path = TASK_MODULES[task]
    try:
        module = import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}'. Ensure PyPOTS is properly installed. Error: {e}")

    if not hasattr(module, model_name):
        available = list_available_models(task)
        raise ValueError(f"Model '{model_name}' not found in task '{task}'. Available models: {available}")
    return getattr(module, model_name)


def get_optimizer_class(optimizer_name: str):
    """Lazily resolve an optimizer class by name.

    Parameters
    ----------
    optimizer_name : str
        The optimizer class name, e.g. "Adam", "AdamW".

    Returns
    -------
    optimizer_class : type
        The resolved optimizer class.
    """
    assert optimizer_name in OPTIMIZER_REGISTRY, (
        f"Unknown optimizer '{optimizer_name}'. Supported optimizers: {list(OPTIMIZER_REGISTRY.keys())}"
    )
    module_path = OPTIMIZER_REGISTRY[optimizer_name].rsplit(".", 1)[0]
    class_name = OPTIMIZER_REGISTRY[optimizer_name].rsplit(".", 1)[1]
    module = import_module(module_path)
    return getattr(module, class_name)


def list_available_models(task: Optional[str] = None) -> dict:
    """List available model names, optionally filtered by task.

    Parameters
    ----------
    task : str or None
        If provided, list only models for this task. Otherwise list all.

    Returns
    -------
    models : dict
        A dict mapping task names to lists of model names.
    """
    tasks_to_list = [task] if task else SUPPORTED_TASKS
    result = {}
    for t in tasks_to_list:
        assert t in SUPPORTED_TASKS, f"Unknown task '{t}'. Supported tasks: {SUPPORTED_TASKS}"
        module_path = TASK_MODULES[t]
        try:
            module = import_module(module_path)
            result[t] = list(getattr(module, "__all__", []))
        except ImportError:
            _get_logger().warning(f"Could not import module for task '{t}', skipping.")
            result[t] = []
    return result


def get_model_init_params(task: str, model_name: str) -> dict:
    """Inspect a model class and return its __init__ parameters with defaults.

    Parameters
    ----------
    task : str
        The task type.
    model_name : str
        The model class name.

    Returns
    -------
    params : dict
        A dict mapping parameter names to their info (annotation, default).
    """
    model_class = get_model_class(task, model_name)
    sig = inspect.signature(model_class)
    params = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        info = {}
        if param.annotation != inspect.Parameter.empty:
            info["type"] = str(param.annotation)
        if param.default != inspect.Parameter.empty:
            default = param.default
            # Convert non-serializable defaults to strings
            if default is None or isinstance(default, (int, float, str, bool, list, dict)):
                info["default"] = default
            else:
                info["default"] = str(default)
        params[name] = info
    return params


def generate_model_config_template(task: str, model_name: str) -> dict:
    """Generate a YAML-serializable config template for a given model.

    Parameters
    ----------
    task : str
        The task type.
    model_name : str
        The model class name.

    Returns
    -------
    template : dict
        A configuration template dictionary.
    """
    params = get_model_init_params(task, model_name)

    model_params = {}
    training_params = {}
    training_keys = {
        "epochs",
        "batch_size",
        "patience",
        "optimizer",
        "num_workers",
        "saving_path",
        "model_saving_strategy",
        "verbose",
        "training_loss",
        "validation_metric",
    }

    for name, info in params.items():
        default_val = info.get("default", f"<{info.get('type', 'required')}>")
        if name in training_keys:
            training_params[name] = default_val
        else:
            model_params[name] = default_val

    template = {
        "task": task,
        "model": {
            "name": model_name,
            **model_params,
        },
        "training": training_params,
        "data": {
            "train_set": "<path_to_train_data.h5>",
            "val_set": "<path_to_val_data.h5>",
            "file_type": "hdf5",
        },
        "device": "cpu",
        "seed": 2024,
    }
    return template
