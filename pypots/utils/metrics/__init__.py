"""Package initialization for pypots.utils.metrics."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.functional.classification import (
    calc_acc,
    calc_binary_classification_metrics,
    calc_pr_auc,
    calc_precision_recall_f1,
    calc_roc_auc,
)
from ...nn.functional.clustering import (
    calc_adjusted_rand_index,
    calc_chs,
    calc_cluster_purity,
    calc_dbs,
    calc_external_cluster_validation_metrics,
    calc_internal_cluster_validation_metrics,
    calc_nmi,
    calc_rand_index,
    calc_silhouette,
)
from ...nn.functional.error import (
    calc_mae,
    calc_mre,
    calc_mse,
    calc_quantile_crps,
    calc_quantile_crps_sum,
    calc_rmse,
)
from ..logging import logger

logger.warning("‼️ `pypots.utils.metrics` is deprecated. Please import from `pypots.nn.functional` instead.")

__all__ = [
    "calc_acc",
    "calc_adjusted_rand_index",
    # classification
    "calc_binary_classification_metrics",
    "calc_chs",
    "calc_cluster_purity",
    "calc_dbs",
    "calc_external_cluster_validation_metrics",
    "calc_internal_cluster_validation_metrics",
    # error
    "calc_mae",
    "calc_mre",
    "calc_mse",
    "calc_nmi",
    "calc_pr_auc",
    "calc_precision_recall_f1",
    "calc_quantile_crps",
    "calc_quantile_crps_sum",
    # clustering
    "calc_rand_index",
    "calc_rmse",
    "calc_roc_auc",
    "calc_silhouette",
]
