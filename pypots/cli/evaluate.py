"""
CLI command to evaluate model predictions against ground truth.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import json
import os

import click

TASK_CHOICES = ["imputation", "classification", "forecasting", "anomaly_detection", "clustering"]

TASK_METRICS = {
    "imputation": ["mse", "mae", "rmse", "mre"],
    "forecasting": ["mse", "mae", "rmse", "mre"],
    "classification": ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"],
    "anomaly_detection": ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"],
    "clustering": ["rand_index", "adjusted_rand_index", "nmi", "cluster_purity", "silhouette", "chs", "dbs"],
}


def _evaluate_imputation_forecasting(task, pred_data, gt_data, metrics_to_compute):
    """Evaluate imputation or forecasting predictions."""
    import numpy as np
    import torch

    from ..nn.functional import calc_mse, calc_mae, calc_rmse, calc_mre
    from ..utils.logging import logger

    pred_key = task  # "imputation" or "forecasting"
    assert pred_key in pred_data, (
        f"Key '{pred_key}' not found in predictions file. Available keys: {list(pred_data.keys())}"
    )
    assert "X_ori" in gt_data, f"Key 'X_ori' not found in ground truth file. Available keys: {list(gt_data.keys())}"

    targets_np = np.asarray(gt_data["X_ori"], dtype=np.float32)

    masks = None
    if "indicating_mask" in gt_data:
        masks = torch.from_numpy(np.asarray(gt_data["indicating_mask"], dtype=np.float32))
        logger.info("Using 'indicating_mask' from ground truth file for evaluation.")
    elif "X" in gt_data:
        # Auto-compute indicating_mask: positions observed in X_ori but artificially masked in X
        X = np.asarray(gt_data["X"], dtype=np.float32)
        indicating_mask = (~np.isnan(targets_np)) & np.isnan(X)
        n_eval_positions = int(indicating_mask.sum())
        n_natural_nan = int(np.isnan(targets_np).sum())
        masks = torch.from_numpy(indicating_mask.astype(np.float32))
        logger.info(
            f"Auto-computed indicating_mask from X and X_ori: "
            f"{n_eval_positions} artificially masked positions will be evaluated "
            f"({n_natural_nan} naturally missing positions excluded)."
        )

    # Replace NaN in targets with 0 at positions where mask is 0 (not evaluated),
    # so metric functions' NaN assertion passes while only masked positions contribute
    targets_np = np.nan_to_num(targets_np, nan=0.0)

    predictions = torch.from_numpy(np.asarray(pred_data[pred_key], dtype=np.float32))
    targets = torch.from_numpy(targets_np)

    metric_funcs = {
        "mse": calc_mse,
        "mae": calc_mae,
        "rmse": calc_rmse,
        "mre": calc_mre,
    }

    results = {}
    for metric_name in metrics_to_compute:
        func = metric_funcs[metric_name]
        value = func(predictions, targets, masks)
        results[metric_name] = float(value)
    return results


def _evaluate_classification(pred_data, gt_data, metrics_to_compute):
    """Evaluate classification or anomaly detection predictions."""
    import numpy as np

    from ..nn.functional import calc_binary_classification_metrics

    prob_key = "classification_proba" if "classification_proba" in pred_data else "classification"
    assert prob_key in pred_data, (
        f"Key 'classification' or 'classification_proba' not found in predictions file. "
        f"Available keys: {list(pred_data.keys())}"
    )
    assert "y" in gt_data, f"Key 'y' not found in ground truth file. Available keys: {list(gt_data.keys())}"

    prob_predictions = np.asarray(pred_data[prob_key], dtype=np.float64)
    targets = np.asarray(gt_data["y"], dtype=np.int64)

    all_metrics = calc_binary_classification_metrics(prob_predictions, targets)

    results = {}
    for metric_name in metrics_to_compute:
        assert metric_name in all_metrics, (
            f"Metric '{metric_name}' not found in classification metrics output. Available: {list(all_metrics.keys())}"
        )
        results[metric_name] = float(all_metrics[metric_name])
    return results


def _evaluate_clustering(pred_data, gt_data, metrics_to_compute):
    """Evaluate clustering predictions."""
    import numpy as np

    from ..nn.functional import (
        calc_external_cluster_validation_metrics,
        calc_internal_cluster_validation_metrics,
    )

    assert "clustering" in pred_data, (
        f"Key 'clustering' not found in predictions file. Available keys: {list(pred_data.keys())}"
    )

    predicted_labels = np.asarray(pred_data["clustering"], dtype=np.int64)

    external_metrics = {"rand_index", "adjusted_rand_index", "nmi", "cluster_purity"}
    internal_metrics = {"silhouette", "chs", "dbs"}
    internal_key_map = {
        "silhouette": "silhouette_score",
        "chs": "calinski_harabasz_score",
        "dbs": "davies_bouldin_score",
    }

    results = {}

    # compute external metrics if any are requested
    requested_external = [m for m in metrics_to_compute if m in external_metrics]
    if requested_external:
        assert "y" in gt_data, (
            f"Key 'y' not found in ground truth file (needed for external clustering metrics). "
            f"Available keys: {list(gt_data.keys())}"
        )
        targets = np.asarray(gt_data["y"], dtype=np.int64)
        ext_results = calc_external_cluster_validation_metrics(predicted_labels, targets)
        for m in requested_external:
            results[m] = float(ext_results[m])

    # compute internal metrics if any are requested
    requested_internal = [m for m in metrics_to_compute if m in internal_metrics]
    if requested_internal:
        assert "X" in pred_data, (
            f"Key 'X' not found in predictions file (needed for internal clustering metrics). "
            f"Available keys: {list(pred_data.keys())}"
        )
        X = np.asarray(pred_data["X"], dtype=np.float64)
        int_results = calc_internal_cluster_validation_metrics(X, predicted_labels)
        for m in requested_internal:
            results[m] = float(int_results[internal_key_map[m]])

    return results


def _print_results(task, results):
    """Print evaluation results in a formatted table."""
    from ..utils.logging import logger

    logger.info(f"Evaluation results for task '{task}':")
    header = f"{'Metric':<30} {'Value':>15}"
    separator = "-" * 46
    print(separator)
    print(header)
    print(separator)
    for metric_name, value in results.items():
        print(f"{metric_name:<30} {value:>15.6f}")
    print(separator)


def _save_results(output_path, results):
    """Save evaluation results as JSON."""
    from ..utils.logging import logger

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {output_path}")


@click.command(name="evaluate", help="Evaluate model predictions against ground truth")
@click.option(
    "--predictions",
    required=True,
    type=click.Path(exists=True),
    help="Path to prediction results H5 file (as saved by predict command)",
)
@click.option("--ground_truth", required=True, type=click.Path(exists=True), help="Path to ground truth data H5 file")
@click.option(
    "--task",
    required=True,
    type=click.Choice(TASK_CHOICES),
    help="Task type for evaluation: imputation, classification, forecasting, anomaly_detection, clustering",
)
@click.option(
    "--metrics",
    default=None,
    type=str,
    help="Comma-separated metric names to compute (default: all applicable metrics for the task)",
)
@click.option(
    "--output",
    default=None,
    type=str,
    help="Path to save evaluation results as JSON (optional; if not given, only prints)",
)
def evaluate(predictions, ground_truth, task, metrics, output):
    """Execute the evaluate command."""
    from ..utils.logging import logger

    # Validate metrics if provided
    if metrics is not None:
        requested = [m.strip() for m in metrics.split(",")]
        valid = TASK_METRICS[task]
        for m in requested:
            assert m in valid, f"Metric '{m}' is not available for task '{task}'. Available metrics: {valid}"

    from ..data.saving.h5 import load_dict_from_h5

    logger.info(f"Loading predictions from {predictions}...")
    pred_data = load_dict_from_h5(predictions)
    logger.info(f"Loading ground truth from {ground_truth}...")
    gt_data = load_dict_from_h5(ground_truth)

    # determine which metrics to compute
    if metrics is not None:
        metrics_to_compute = [m.strip() for m in metrics.split(",")]
    else:
        metrics_to_compute = TASK_METRICS[task]

    logger.info(f"Computing metrics for task '{task}': {metrics_to_compute}")

    if task in ("imputation", "forecasting"):
        results = _evaluate_imputation_forecasting(task, pred_data, gt_data, metrics_to_compute)
    elif task in ("classification", "anomaly_detection"):
        results = _evaluate_classification(pred_data, gt_data, metrics_to_compute)
    elif task == "clustering":
        results = _evaluate_clustering(pred_data, gt_data, metrics_to_compute)
    else:
        raise ValueError(f"Unknown task type: {task}")

    _print_results(task, results)

    if output is not None:
        _save_results(output, results)
