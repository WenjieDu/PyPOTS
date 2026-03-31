"""
CLI command to evaluate model predictions against ground truth.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import json
import os
from argparse import Namespace

import numpy as np
import torch

from .base import BaseCommand
from ..utils.logging import logger

TASK_CHOICES = ["imputation", "classification", "forecasting", "anomaly_detection", "clustering"]

TASK_METRICS = {
    "imputation": ["mse", "mae", "rmse", "mre"],
    "forecasting": ["mse", "mae", "rmse", "mre"],
    "classification": ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"],
    "anomaly_detection": ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"],
    "clustering": ["rand_index", "adjusted_rand_index", "nmi", "cluster_purity", "silhouette", "chs", "dbs"],
}


def evaluate_command_factory(args: Namespace):
    return EvaluateCommand(
        args.predictions,
        args.ground_truth,
        args.task,
        args.metrics,
        args.output,
    )


class EvaluateCommand(BaseCommand):
    """CLI command for evaluating model predictions against ground truth data.

    Examples
    --------
    $ pypots-cli evaluate --predictions ./predictions.h5 --ground_truth ./test.h5 --task imputation
    $ pypots-cli evaluate --predictions ./pred.h5 --ground_truth ./gt.h5 --task classification --metrics accuracy,f1
    $ pypots-cli evaluate --predictions ./pred.h5 --ground_truth ./gt.h5 --task clustering --output ./eval_results.json

    """

    @staticmethod
    def register_subcommand(parser):
        sub_parser = parser.add_parser(
            "evaluate",
            help="Evaluate model predictions against ground truth",
        )
        sub_parser.add_argument(
            "--predictions",
            dest="predictions",
            type=str,
            required=True,
            help="Path to prediction results H5 file (as saved by predict command)",
        )
        sub_parser.add_argument(
            "--ground_truth",
            "--ground-truth",
            dest="ground_truth",
            type=str,
            required=True,
            help="Path to ground truth data H5 file",
        )
        sub_parser.add_argument(
            "--task",
            dest="task",
            type=str,
            required=True,
            choices=TASK_CHOICES,
            help="Task type for evaluation: imputation, classification, forecasting, anomaly_detection, clustering",
        )
        sub_parser.add_argument(
            "--metrics",
            dest="metrics",
            type=str,
            default=None,
            help="Comma-separated metric names to compute (default: all applicable metrics for the task)",
        )
        sub_parser.add_argument(
            "--output",
            dest="output",
            type=str,
            default=None,
            help="Path to save evaluation results as JSON (optional; if not given, only prints)",
        )
        sub_parser.set_defaults(func=evaluate_command_factory)

    def __init__(
        self,
        predictions: str,
        ground_truth: str,
        task: str,
        metrics: str,
        output: str,
    ):
        self._predictions = predictions
        self._ground_truth = ground_truth
        self._task = task
        self._metrics = metrics
        self._output = output

    def checkup(self):
        """Run some checks on the arguments to avoid error usages."""
        assert os.path.exists(self._predictions), (
            f"Predictions file not found: {self._predictions}"
        )
        assert os.path.exists(self._ground_truth), (
            f"Ground truth file not found: {self._ground_truth}"
        )

        if self._metrics is not None:
            requested = [m.strip() for m in self._metrics.split(",")]
            valid = TASK_METRICS[self._task]
            for m in requested:
                assert m in valid, (
                    f"Metric '{m}' is not available for task '{self._task}'. "
                    f"Available metrics: {valid}"
                )

    def _evaluate_imputation_forecasting(self, pred_data, gt_data, metrics_to_compute):
        """Evaluate imputation or forecasting predictions."""
        from ..nn.functional import calc_mse, calc_mae, calc_rmse, calc_mre

        pred_key = self._task  # "imputation" or "forecasting"
        assert pred_key in pred_data, (
            f"Key '{pred_key}' not found in predictions file. Available keys: {list(pred_data.keys())}"
        )
        assert "X_ori" in gt_data, (
            f"Key 'X_ori' not found in ground truth file. Available keys: {list(gt_data.keys())}"
        )

        predictions = torch.from_numpy(np.asarray(pred_data[pred_key], dtype=np.float32))
        targets = torch.from_numpy(np.asarray(gt_data["X_ori"], dtype=np.float32))

        masks = None
        if "indicating_mask" in gt_data:
            masks = torch.from_numpy(np.asarray(gt_data["indicating_mask"], dtype=np.float32))

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

    def _evaluate_classification(self, pred_data, gt_data, metrics_to_compute):
        """Evaluate classification or anomaly detection predictions."""
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
                f"Metric '{metric_name}' not found in classification metrics output. "
                f"Available: {list(all_metrics.keys())}"
            )
            results[metric_name] = float(all_metrics[metric_name])
        return results

    def _evaluate_clustering(self, pred_data, gt_data, metrics_to_compute):
        """Evaluate clustering predictions."""
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

    def _print_results(self, results):
        """Print evaluation results in a formatted table."""
        logger.info(f"Evaluation results for task '{self._task}':")
        header = f"{'Metric':<30} {'Value':>15}"
        separator = "-" * 46
        print(separator)
        print(header)
        print(separator)
        for metric_name, value in results.items():
            print(f"{metric_name:<30} {value:>15.6f}")
        print(separator)

    def _save_results(self, results):
        """Save evaluation results as JSON."""
        output_dir = os.path.dirname(self._output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self._output, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to {self._output}")

    def run(self):
        """Execute the evaluate command."""
        self.checkup()

        from ..data.saving.h5 import load_dict_from_h5

        logger.info(f"Loading predictions from {self._predictions}...")
        pred_data = load_dict_from_h5(self._predictions)
        logger.info(f"Loading ground truth from {self._ground_truth}...")
        gt_data = load_dict_from_h5(self._ground_truth)

        # determine which metrics to compute
        if self._metrics is not None:
            metrics_to_compute = [m.strip() for m in self._metrics.split(",")]
        else:
            metrics_to_compute = TASK_METRICS[self._task]

        logger.info(f"Computing metrics for task '{self._task}': {metrics_to_compute}")

        if self._task in ("imputation", "forecasting"):
            results = self._evaluate_imputation_forecasting(pred_data, gt_data, metrics_to_compute)
        elif self._task in ("classification", "anomaly_detection"):
            results = self._evaluate_classification(pred_data, gt_data, metrics_to_compute)
        elif self._task == "clustering":
            results = self._evaluate_clustering(pred_data, gt_data, metrics_to_compute)
        else:
            raise ValueError(f"Unknown task type: {self._task}")

        self._print_results(results)

        if self._output is not None:
            self._save_results(results)
