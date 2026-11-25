"""Package initialization for pypots.nn.functional."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .classification import (
    calc_acc,
    calc_binary_classification_metrics,
    calc_pr_auc,
    calc_precision_recall_f1,
    calc_roc_auc,
)
from .clustering import (
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
from .cuda import autocast
from .error import (
    calc_mae,
    calc_mre,
    calc_mse,
    calc_quantile_crps,
    calc_quantile_crps_sum,
    calc_rmse,
)
from .gathering import gather_listed_dicts
from .normalization import nonstationary_denorm, nonstationary_norm

__all__ = [
    # cuda functions
    "autocast",
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
    # gathering functions
    "gather_listed_dicts",
    "nonstationary_denorm",
    # normalization functions
    "nonstationary_norm",
]
