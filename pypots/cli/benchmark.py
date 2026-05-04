"""
CLI command for benchmarking multiple models on the same dataset.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import inspect
import json

import click

from .utils import load_config, merge_config_with_overrides, get_model_class


# Tasks that use regression-style metrics (mse, mae, rmse, mre)
REGRESSION_METRIC_TASKS = {"imputation", "forecasting"}
# Result key produced by model.predict() for each task
TASK_PREDICTION_KEY = {
    "imputation": "imputation",
    "forecasting": "forecasting",
}


def _get_regression_metric_funcs():
    """Lazy-load metric functions to avoid importing torch/numpy at module level."""
    from ..nn.functional import calc_mse, calc_mae, calc_rmse, calc_mre

    return {
        "mse": calc_mse,
        "mae": calc_mae,
        "rmse": calc_rmse,
        "mre": calc_mre,
    }


def _compute_metrics(task: str, prediction_results: dict, test_set: str, metrics_list: list) -> dict:
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
    import numpy as np
    import torch

    from ..data.saving.h5 import load_dict_from_h5

    metrics = {}

    if task in REGRESSION_METRIC_TASKS:
        regression_metric_funcs = _get_regression_metric_funcs()
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

        from ..utils.logging import logger as _logger

        for metric_name in metrics_list:
            metric_name_lower = metric_name.lower()
            if metric_name_lower not in regression_metric_funcs:
                _logger.warning(f"Unknown metric '{metric_name}' for task '{task}', skipping.")
                continue
            func = regression_metric_funcs[metric_name_lower]
            value = func(predictions, X_ori, indicating_mask)
            metrics[metric_name_lower] = float(value)

    elif task in ("classification", "anomaly_detection"):
        from ..nn.functional import calc_binary_classification_metrics

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
        from ..nn.functional import calc_external_cluster_validation_metrics

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
        from ..utils.logging import logger as _logger

        _logger.warning(f"Metric computation not implemented for task '{task}'. Returning empty metrics.")

    return metrics


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

    from ..utils.logging import logger

    logger.info(f"Benchmark results saved to {output_path}")


def _checkup(config_path: str):
    """Validate the benchmark configuration before running."""
    cfg = load_config(config_path)

    task = cfg.get("task")
    assert task is not None, "Benchmark config must specify a 'task' field."

    models = cfg.get("models")
    assert models and len(models) > 0, "Benchmark config must specify at least one model in 'models'."

    data = cfg.get("data")
    assert data is not None, "Benchmark config must specify a 'data' section."
    assert "train_set" in data, "Benchmark config 'data' must include 'train_set'."
    assert "test_set" in data, "Benchmark config 'data' must include 'test_set'."

    # Validate that all model classes can be resolved
    for model_entry in models:
        name = model_entry.get("name")
        assert name is not None, "Each model entry must have a 'name' field."
        get_model_class(task, name)


@click.command(name="benchmark", help="Benchmark multiple models on the same dataset and compare metrics")
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to YAML/JSON benchmark configuration file")
@click.option("--device", default=None, type=str, help="Override device for all models (e.g. cpu, cuda:0)")
@click.option("--seed", type=int, default=None, help="Override random seed for reproducibility")
@click.option("--output", default=None, type=str, help="Override output file path for benchmark results JSON")
def benchmark(config, device, seed, output):
    """Execute the benchmark: train each model, predict, compute metrics, and report results."""
    import numpy as np

    from ..utils.logging import logger
    from ..utils.random import set_random_seed

    # Load config and merge CLI overrides
    cfg = load_config(config)
    cfg = merge_config_with_overrides(cfg, {"device": device, "seed": seed, "output": output})

    task = cfg["task"]
    models_cfg = cfg["models"]
    data_cfg = cfg["data"]
    metrics_list = cfg.get("metrics", ["mse", "mae"])
    output_path = cfg.get("output")
    resolved_device = cfg.get("device")
    resolved_seed = cfg.get("seed", 2024)

    # Set random seed
    set_random_seed(resolved_seed)
    logger.info(f"Random seed set to {resolved_seed}")

    # Validate all models exist before starting any training
    _checkup(config)
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
        if resolved_device is not None:
            kwargs["device"] = resolved_device

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
        model_metrics = _compute_metrics(task, results, test_set, metrics_list)
        all_results[model_name] = model_metrics
        logger.info(f"Model {model_name} metrics: {model_metrics}")

    # Print comparison table
    _print_comparison_table(task, all_results, metrics_list)

    # Save results to JSON if output path is specified
    if output_path:
        _save_results(output_path, task, all_results)
