"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .classification import (
    calc_binary_classification_metrics,
    calc_precision_recall_f1,
    calc_pr_auc,
    calc_roc_auc,
    calc_acc,
)
from .clustering import (
    calc_rand_index,
    calc_adjusted_rand_index,
    calc_cluster_purity,
    calc_nmi,
    calc_chs,
    calc_dbs,
    calc_silhouette,
    calc_internal_cluster_validation_metrics,
    calc_external_cluster_validation_metrics,
)
from .cuda import autocast
from .error import (
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mre,
    calc_quantile_crps,
    calc_quantile_crps_sum,
)
from .gathering import gather_listed_dicts
from .normalization import nonstationary_norm, nonstationary_denorm

__all__ = [
    # cuda functions
    "autocast",
    # gathering functions
    "gather_listed_dicts",
    # normalization functions
    "nonstationary_norm",
    "nonstationary_denorm",
    # error
    "calc_mae",
    "calc_mse",
    "calc_rmse",
    "calc_mre",
    "calc_quantile_crps",
    "calc_quantile_crps_sum",
    # classification
    "calc_binary_classification_metrics",
    "calc_precision_recall_f1",
    "calc_pr_auc",
    "calc_roc_auc",
    "calc_acc",
    # clustering
    "calc_rand_index",
    "calc_adjusted_rand_index",
    "calc_cluster_purity",
    "calc_nmi",
    "calc_chs",
    "calc_dbs",
    "calc_silhouette",
    "calc_internal_cluster_validation_metrics",
    "calc_external_cluster_validation_metrics",
]
