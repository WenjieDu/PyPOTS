"""
CLI command for benchmarking multiple models on the same dataset.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect
import json
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from .base import BaseCommand
from .utils import load_config, merge_config_with_args, get_model_class
from ..data.saving.h5 import load_dict_from_h5
from ..nn.functional import (
    calc_mse,
    calc_mae,
    calc_rmse,
    calc_mre,
    calc_binary_classification_metrics,
    calc_external_cluster_validation_metrics,
)
from ..utils.logging import logger
from ..utils.random import set_random_seed

# Tasks that use regression-style metrics (mse, mae, rmse, mre)
REGRESSION_METRIC_TASKS = {"imputation", "forecasting"}
# Mapping from metric name to its computation function
REGRESSION_METRIC_FUNCS = {
    "mse": calc_mse,
    "mae": calc_mae,
    "rmse": calc_rmse,
    "mre": calc_mre,
}
# Result key produced by model.predict() for each task
TASK_PREDICTION_KEY = {
    "imputation": "imputation",
    "forecasting": "forecasting",
}


def benchmark_command_factory(args: Namespace):
    return BenchmarkCommand(
        config_path=args.config,
        device=getattr(args, "device", None),
        seed=getattr(args, "seed", None),
        output=getattr(args, "output", None),
    )


class BenchmarkCommand(BaseCommand):
    """CLI command for benchmarking multiple models on the same dataset.

    Examples
    --------
    $ pypots-cli benchmark --config benchmark_config.yaml
    $ pypots-cli benchmark --config benchmark_config.yaml --device cpu --seed 2024
    $ pypots-cli benchmark --config benchmark_config.yaml --output results.json
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "benchmark",
            help="Benchmark multiple models on the same dataset and compare metrics",
            allow_abbrev=True,
        )
        sub_parser.add_argument(
            "--config",
            dest="config",
            type=str,
            required=True,
            help="Path to YAML/JSON benchmark configuration file",
        )
        sub_parser.add_argument(
            "--device",
            dest="device",
            type=str,
            required=False,
            help="Override device for all models (e.g. cpu, cuda:0)",
        )
        sub_parser.add_argument(
            "--seed",
            dest="seed",
            type=int,
            required=False,
            help="Override random seed for reproducibility",
        )
        sub_parser.add_argument(
            "--output",
            dest="output",
            type=str,
            required=False,
            help="Override output file path for benchmark results JSON",
        )
        sub_parser.set_defaults(func=benchmark_command_factory)

    def __init__(self, config_path: str, device: str = None, seed: int = None, output: str = None):
        self._config_path = config_path
        self._device = device
        self._seed = seed
        self._output = output

    def checkup(self):
        """Validate the benchmark configuration before running."""
        config = load_config(self._config_path)

        task = config.get("task")
        assert task is not None, "Benchmark config must specify a 'task' field."

        models = config.get("models")
        assert models and len(models) > 0, "Benchmark config must specify at least one model in 'models'."

        data = config.get("data")
        assert data is not None, "Benchmark config must specify a 'data' section."
        assert "train_set" in data, "Benchmark config 'data' must include 'train_set'."
        assert "test_set" in data, "Benchmark config 'data' must include 'test_set'."

        # Validate that all model classes can be resolved
        for model_entry in models:
            name = model_entry.get("name")
            assert name is not None, "Each model entry must have a 'name' field."
            get_model_class(task, name)

    def run(self):
        """Execute the benchmark: train each model, predict, compute metrics, and report results."""
        # Load config and merge CLI overrides
        config = load_config(self._config_path)
        config = merge_config_with_args(config, Namespace(device=self._device, seed=self._seed, output=self._output),
                                        ["device", "seed", "output"])

        task = config["task"]
        models_cfg = config["models"]
        data_cfg = config["data"]
        metrics_list = config.get("metrics", ["mse", "mae"])
        output_path = config.get("output")
        device = config.get("device")
        seed = config.get("seed", 2024)

        # Set random seed
        set_random_seed(seed)
        logger.info(f"Random seed set to {seed}")

        # Validate all models exist before starting any training
        self.checkup()
        logger.info(f"Benchmark starting — task: {task}, models: {[m['name'] for m in models_cfg]}")

        # Load data paths
        train_set = data_cfg["train_set"]
        val_set = data_cfg.get("val_set")
        test_set = data_cfg["test_set"]

        # Collect results for each model
        all_results = {}

        for model_entry in models_cfg:
            model_name = model_entry["name"]
            model_params = model_entry.get("params", {})

            # Resolve model class
            model_class = get_model_class(task, model_name)

            # Build kwargs: merge model params with device override
            kwargs = dict(model_params)
            if device is not None:
                kwargs["device"] = device

            # Filter kwargs to only those accepted by the model's __init__
            sig = inspect.signature(model_class.__init__)
            accepted_params = set(sig.parameters.keys()) - {"self"}
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_keyword:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
                skipped = set(kwargs.keys()) - set(filtered_kwargs.keys())
                if skipped:
                    logger.warning(
                        f"Skipping parameters not accepted by {model_name}: {skipped}"
                    )
                kwargs = filtered_kwargs

            logger.info(f"Training model {model_name}...")
            model = model_class(**kwargs)
            model.fit(train_set=train_set, val_set=val_set)

            logger.info(f"Predicting with model {model_name}...")
            results = model.predict(test_set)

            # Compute metrics
            model_metrics = self._compute_metrics(task, results, test_set, metrics_list)
            all_results[model_name] = model_metrics
            logger.info(f"Model {model_name} metrics: {model_metrics}")

        # Print comparison table
        self._print_comparison_table(task, all_results, metrics_list)

        # Save results to JSON if output path is specified
        if output_path:
            self._save_results(output_path, task, all_results)

    def _compute_metrics(self, task: str, prediction_results: dict, test_set: str, metrics_list: list) -> dict:
        """Compute requested metrics for a single model's predictions.

        Parameters
        ----------
        task : str
            The task type.
        prediction_results : dict
            The dict returned by model.predict().
        test_set : str
            Path to the test set H5 file (used to load ground truth).
        metrics_list : list
            List of metric names to compute.

        Returns
        -------
        metrics : dict
            Mapping from metric name to its computed value.
        """
        metrics = {}

        if task in REGRESSION_METRIC_TASKS:
            # Load ground truth from the test set
            test_data = load_dict_from_h5(test_set)
            X_ori = test_data["X_ori"]
            indicating_mask = test_data.get("indicating_mask")

            # Get predictions
            pred_key = TASK_PREDICTION_KEY.get(task, task)
            predictions = prediction_results[pred_key]

            # Convert to torch tensors
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions).float()
            if isinstance(X_ori, np.ndarray):
                X_ori = torch.from_numpy(X_ori).float()
            if indicating_mask is not None and isinstance(indicating_mask, np.ndarray):
                indicating_mask = torch.from_numpy(indicating_mask).float()

            for metric_name in metrics_list:
                metric_name_lower = metric_name.lower()
                if metric_name_lower not in REGRESSION_METRIC_FUNCS:
                    logger.warning(f"Unknown metric '{metric_name}' for task '{task}', skipping.")
                    continue
                func = REGRESSION_METRIC_FUNCS[metric_name_lower]
                value = func(predictions, X_ori, indicating_mask)
                metrics[metric_name_lower] = float(value)

        elif task in ("classification", "anomaly_detection"):
            prob_predictions = prediction_results.get("classification", prediction_results.get("anomaly_detection"))
            test_data = load_dict_from_h5(test_set)
            targets = test_data.get("y", test_data.get("labels"))

            if isinstance(prob_predictions, torch.Tensor):
                prob_predictions = prob_predictions.numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()

            cls_metrics = calc_binary_classification_metrics(prob_predictions, targets)
            # Filter to only requested metrics, or return all if none specifically match
            for metric_name in metrics_list:
                metric_name_lower = metric_name.lower()
                if metric_name_lower in cls_metrics:
                    metrics[metric_name_lower] = float(cls_metrics[metric_name_lower])
            if not metrics:
                metrics = {k: float(v) for k, v in cls_metrics.items()}

        elif task == "clustering":
            cluster_predictions = prediction_results.get("clustering")
            test_data = load_dict_from_h5(test_set)
            targets = test_data.get("y", test_data.get("labels"))

            if isinstance(cluster_predictions, torch.Tensor):
                cluster_predictions = cluster_predictions.numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()

            cluster_metrics = calc_external_cluster_validation_metrics(cluster_predictions, targets)
            for metric_name in metrics_list:
                metric_name_lower = metric_name.lower()
                if metric_name_lower in cluster_metrics:
                    metrics[metric_name_lower] = float(cluster_metrics[metric_name_lower])
            if not metrics:
                metrics = {k: float(v) for k, v in cluster_metrics.items()}

        else:
            logger.warning(f"Metric computation not implemented for task '{task}'. Returning empty metrics.")

        return metrics

    @staticmethod
    def _print_comparison_table(task: str, all_results: dict, metrics_list: list):
        """Print a formatted comparison table of benchmark results.

        Parameters
        ----------
        task : str
            The task type.
        all_results : dict
            Mapping from model name to its metrics dict.
        metrics_list : list
            Ordered list of metric names for column headers.
        """
        # Determine which metrics actually have values
        available_metrics = []
        for m in metrics_list:
            m_lower = m.lower()
            if any(m_lower in model_metrics for model_metrics in all_results.values()):
                available_metrics.append(m_lower)

        if not available_metrics:
            # Fall back: collect all metric keys across models
            for model_metrics in all_results.values():
                for k in model_metrics:
                    if k not in available_metrics:
                        available_metrics.append(k)

        # Column widths
        model_col_width = max(15, max((len(name) for name in all_results), default=5) + 2)
        metric_col_width = 10

        header_line = f"{'Model':<{model_col_width}}"
        for m in available_metrics:
            header_line += f"| {m.upper():<{metric_col_width}}"

        separator_width = model_col_width + (metric_col_width + 2) * len(available_metrics)

        print("\n" + "=" * separator_width)
        print(f"Benchmark Results - Task: {task}")
        print("=" * separator_width)
        print(header_line)
        print("-" * separator_width)

        for model_name, model_metrics in all_results.items():
            row = f"{model_name:<{model_col_width}}"
            for m in available_metrics:
                value = model_metrics.get(m)
                if value is not None:
                    row += f"| {value:<{metric_col_width}.4f}"
                else:
                    row += f"| {'N/A':<{metric_col_width}}"
            print(row)

        print("=" * separator_width + "\n")

    @staticmethod
    def _save_results(output_path: str, task: str, all_results: dict):
        """Save benchmark results to a JSON file.

        Parameters
        ----------
        output_path : str
            File path for the output JSON.
        task : str
            The task type.
        all_results : dict
            Mapping from model name to its metrics dict.
        """
        output = {
            "task": task,
            "results": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Benchmark results saved to {output_path}")
