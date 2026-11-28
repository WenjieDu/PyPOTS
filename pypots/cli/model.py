"""
CLI command for model management (list, describe, inspect, config).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import json
import os

import click

from .base import safe_print

TASK_CHOICES = ["imputation", "classification", "forecasting", "anomaly_detection", "clustering", "representation"]


def _save_yaml(data: dict, path: str):
    """Save a dictionary as a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to save YAML config files. Install it with: pip install pyyaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    from ..utils.logging import logger

    logger.info(f"Config template saved to {path}")


def _format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


@click.group(name="model", help="Model management operations: list, describe, inspect, config")
def model():
    """Model management operations: list, describe, inspect, config."""


@model.command(name="list", help="List available models, optionally filtered by task")
@click.option("--task", default=None, type=click.Choice(TASK_CHOICES), help="Task type to filter models")
def model_list(task):
    """List available models, optionally filtered by task."""
    from .utils import SUPPORTED_TASKS, list_available_models

    models = list_available_models(task)

    safe_print("\n" + "=" * 70)
    safe_print(f"{'Task':<25} {'Count':<8} {'Models'}")
    safe_print("=" * 70)
    for task_name in SUPPORTED_TASKS:
        if task_name not in models:
            continue
        model_list_items = models[task_name]
        count = len(model_list_items)
        names_str = ", ".join(model_list_items) if model_list_items else "(none)"
        safe_print(f"{task_name:<25} {count:<8} {names_str}")
    safe_print("=" * 70 + "\n")


@model.command(name="describe", help="Show detailed information about a specific model")
@click.option("--name", required=True, type=str, help="Model class name (e.g. SAITS, BRITS)")
@click.option("--task", required=True, type=click.Choice(TASK_CHOICES), help="Task type to specify the model")
def model_describe(name, task):
    """Show detailed information about a specific model."""
    from .utils import get_model_class, get_model_init_params

    model_class = get_model_class(task, name)

    # Print model docstring
    safe_print("\n" + "=" * 70)
    safe_print(f"Model: {name}  |  Task: {task}")
    safe_print("=" * 70)

    docstring = model_class.__doc__
    if docstring:
        safe_print("\nDescription:")
        safe_print(docstring)
    else:
        safe_print("\n(No docstring available)")

    # Print __init__ parameters
    params = get_model_init_params(task, name)
    if params:
        safe_print("-" * 70)
        safe_print(f"{'Parameter':<25} {'Type':<30} {'Default'}")
        safe_print("-" * 70)
        for param_name, info in params.items():
            type_str = info.get("type", "-")
            default_str = repr(info["default"]) if "default" in info else "(required)"
            safe_print(f"{param_name:<25} {type_str:<30} {default_str}")
        safe_print("-" * 70 + "\n")
    else:
        safe_print("(No __init__ parameters found)\n")


@model.command(name="inspect", help="Inspect a saved .pypots model file")
@click.option("--path", required=True, type=click.Path(exists=True), help="Path to a saved .pypots model file")
def model_inspect(path):
    """Inspect a saved .pypots model file."""
    try:
        import torch
    except ImportError:
        from ..utils.logging import logger

        logger.error("PyTorch is required to inspect model files. Install it with: pip install torch")
        return

    try:
        file_size = os.path.getsize(path)
        checkpoint = torch.load(path, map_location="cpu")

        safe_print("\n" + "=" * 70)
        safe_print(f"Model file: {path}")
        safe_print(f"File size: {_format_file_size(file_size)}")
        safe_print("=" * 70)

        # Print metadata
        model_class = checkpoint.get("model_class")
        pypots_version = checkpoint.get("pypots_version")
        save_timestamp = checkpoint.get("save_timestamp")
        hyperparameters = checkpoint.get("hyperparameters")

        if model_class or pypots_version or save_timestamp:
            safe_print("\nMetadata:")
            safe_print("-" * 70)
            if model_class:
                safe_print(f"  Model class:    {model_class}")
            if pypots_version:
                safe_print(f"  PyPOTS version: {pypots_version}")
            if save_timestamp:
                safe_print(f"  Saved at:       {save_timestamp}")

        if hyperparameters:
            safe_print(f"\nHyperparameters ({len(hyperparameters)} entries):")
            safe_print("-" * 70)
            for k, v in sorted(hyperparameters.items()):
                safe_print(f"  {k}: {v}")

        # Print top-level keys
        meta_keys = {"model_state_dict", "pypots_version", "model_class", "hyperparameters", "save_timestamp"}
        other_keys = [k for k in checkpoint if k not in meta_keys]
        if other_keys:
            safe_print(f"\nOther checkpoint keys: {other_keys}")

        # Print model state dict layer info
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            total_params = sum(t.numel() for t in state_dict.values() if hasattr(t, "numel"))
            safe_print(f"\nModel state dict ({len(state_dict)} layers, {total_params:,} parameters):")
            safe_print("-" * 70)
            safe_print(f"{'Layer Name':<50} {'Shape'}")
            safe_print("-" * 70)
            for layer_name, tensor in state_dict.items():
                shape_str = str(list(tensor.shape)) if hasattr(tensor, "shape") else str(type(tensor))
                safe_print(f"{layer_name:<50} {shape_str}")
            safe_print("-" * 70)

        safe_print()

    except Exception as e:  # noqa: BLE001 - report checkpoint inspection failures at the CLI boundary
        from ..utils.logging import logger

        logger.error(f"Failed to inspect model file '{path}': {e}")


@model.command(name="config", help="Generate a template configuration file for a model")
@click.option("--name", required=True, type=str, help="Model class name (e.g. SAITS, BRITS)")
@click.option("--task", required=True, type=click.Choice(TASK_CHOICES), help="Task type to specify the model")
@click.option(
    "--output", default=None, type=str, help="Output file path for generated config (supports .yaml, .yml, .json)"
)
def model_config(name, task, output):
    """Generate a template configuration file for a model."""
    from .utils import generate_model_config_template

    template = generate_model_config_template(task, name)

    if output:
        ext = os.path.splitext(output)[1].lower()
        if ext in (".yaml", ".yml"):
            _save_yaml(template, output)
        elif ext == ".json":
            with open(output, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, default=str)

            from ..utils.logging import logger

            logger.info(f"Config template saved to {output}")
        else:
            raise ValueError(f"Unsupported output format '{ext}'. Use .yaml, .yml, or .json.")
    else:
        # Print to stdout: prefer YAML, fall back to JSON
        try:
            import yaml

            safe_print(yaml.dump(template, default_flow_style=False, sort_keys=False, allow_unicode=False))
        except ImportError:
            safe_print(json.dumps(template, indent=2, default=str))
