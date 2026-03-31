"""
CLI command for model management (list, describe, inspect, config).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import json
import os
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from ..utils.logging import logger


def model_command_factory(args: Namespace):
    return ModelCommand(
        action=args.action,
        task=getattr(args, "task", None),
        name=getattr(args, "name", None),
        path=getattr(args, "path", None),
        output=getattr(args, "output", None),
    )


class ModelCommand(BaseCommand):
    """CLI command for model management operations: list, describe, inspect, config.

    Examples
    --------
    $ pypots-cli model list
    $ pypots-cli model list --task imputation
    $ pypots-cli model describe --name SAITS --task imputation
    $ pypots-cli model inspect --path ./model.pypots
    $ pypots-cli model config --name SAITS --task imputation --output saits_config.yaml
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "model",
            help="Model management operations: list, describe, inspect, config",
            allow_abbrev=True,
        )

        sub_parser.add_argument(
            "action",
            type=str,
            choices=["list", "describe", "inspect", "config"],
            help="The model management action to perform",
        )
        sub_parser.add_argument(
            "--task",
            dest="task",
            type=str,
            required=False,
            choices=[
                "imputation", "classification", "forecasting",
                "anomaly_detection", "clustering", "representation",
            ],
            help="Task type to filter or specify models",
        )
        sub_parser.add_argument(
            "--name",
            dest="name",
            type=str,
            required=False,
            help="Model class name (e.g. SAITS, BRITS)",
        )
        sub_parser.add_argument(
            "--path",
            dest="path",
            type=str,
            required=False,
            help="Path to a saved .pypots model file",
        )
        sub_parser.add_argument(
            "--output",
            dest="output",
            type=str,
            required=False,
            help="Output file path for generated config (supports .yaml, .yml, .json)",
        )

        sub_parser.set_defaults(func=model_command_factory)

    def __init__(self, action: str, task: str = None, name: str = None, path: str = None, output: str = None):
        self._action = action
        self._task = task
        self._name = name
        self._path = path
        self._output = output

    def checkup(self):
        """Validate arguments based on the chosen action."""
        if self._action == "describe":
            if not self._name or not self._task:
                raise ValueError("Action 'describe' requires both --name and --task arguments.")
        elif self._action == "inspect":
            if not self._path:
                raise ValueError("Action 'inspect' requires the --path argument.")
            if not os.path.exists(self._path):
                raise FileNotFoundError(f"Model file not found: {self._path}")
        elif self._action == "config":
            if not self._name or not self._task:
                raise ValueError("Action 'config' requires both --name and --task arguments.")

    def run(self):
        """Execute the chosen model management action."""
        self.checkup()

        if self._action == "list":
            self._run_list()
        elif self._action == "describe":
            self._run_describe()
        elif self._action == "inspect":
            self._run_inspect()
        elif self._action == "config":
            self._run_config()

    def _run_list(self):
        """List available models, optionally filtered by task."""
        from .utils import list_available_models, SUPPORTED_TASKS

        models = list_available_models(self._task)

        print("\n" + "=" * 70)
        print(f"{'Task':<25} {'Count':<8} {'Models'}")
        print("=" * 70)
        for task_name in SUPPORTED_TASKS:
            if task_name not in models:
                continue
            model_list = models[task_name]
            count = len(model_list)
            names_str = ", ".join(model_list) if model_list else "(none)"
            print(f"{task_name:<25} {count:<8} {names_str}")
        print("=" * 70 + "\n")

    def _run_describe(self):
        """Show detailed information about a specific model."""
        from .utils import get_model_class, get_model_init_params

        model_class = get_model_class(self._task, self._name)

        # Print model docstring
        print("\n" + "=" * 70)
        print(f"Model: {self._name}  |  Task: {self._task}")
        print("=" * 70)

        docstring = model_class.__doc__
        if docstring:
            print("\nDescription:")
            print(docstring)
        else:
            print("\n(No docstring available)")

        # Print __init__ parameters
        params = get_model_init_params(self._task, self._name)
        if params:
            print("-" * 70)
            print(f"{'Parameter':<25} {'Type':<30} {'Default'}")
            print("-" * 70)
            for param_name, info in params.items():
                type_str = info.get("type", "-")
                default_str = repr(info["default"]) if "default" in info else "(required)"
                print(f"{param_name:<25} {type_str:<30} {default_str}")
            print("-" * 70 + "\n")
        else:
            print("(No __init__ parameters found)\n")

    def _run_inspect(self):
        """Inspect a saved .pypots model file."""
        try:
            import torch
        except ImportError:
            logger.error("PyTorch is required to inspect model files. Install it with: pip install torch")
            return

        try:
            file_size = os.path.getsize(self._path)
            checkpoint = torch.load(self._path, map_location="cpu")

            print("\n" + "=" * 70)
            print(f"Model file: {self._path}")
            print(f"File size: {self._format_file_size(file_size)}")
            print("=" * 70)

            # Print top-level keys
            print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

            # Print model state dict layer info
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                print(f"\nModel state dict ({len(state_dict)} layers):")
                print("-" * 70)
                print(f"{'Layer Name':<50} {'Shape'}")
                print("-" * 70)
                for layer_name, tensor in state_dict.items():
                    shape_str = str(list(tensor.shape)) if hasattr(tensor, "shape") else str(type(tensor))
                    print(f"{layer_name:<50} {shape_str}")
                print("-" * 70)

            print()

        except Exception as e:
            logger.error(f"Failed to inspect model file '{self._path}': {e}")

    def _run_config(self):
        """Generate a template configuration file for a model."""
        from .utils import generate_model_config_template

        template = generate_model_config_template(self._task, self._name)

        if self._output:
            ext = os.path.splitext(self._output)[1].lower()
            if ext in (".yaml", ".yml"):
                self._save_yaml(template, self._output)
            elif ext == ".json":
                with open(self._output, "w") as f:
                    json.dump(template, f, indent=2, default=str)
                logger.info(f"Config template saved to {self._output}")
            else:
                raise ValueError(
                    f"Unsupported output format '{ext}'. Use .yaml, .yml, or .json."
                )
        else:
            # Print to stdout: prefer YAML, fall back to JSON
            try:
                import yaml
                print(yaml.dump(template, default_flow_style=False, sort_keys=False))
            except ImportError:
                print(json.dumps(template, indent=2, default=str))

    @staticmethod
    def _save_yaml(data: dict, path: str):
        """Save a dictionary as a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to save YAML config files. Install it with: pip install pyyaml"
            )
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config template saved to {path}")

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format a file size in bytes to a human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
