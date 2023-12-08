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
    # deprecated
    cal_binary_classification_metrics,
    cal_precision_recall_f1,
    cal_pr_auc,
    cal_roc_auc,
    cal_acc,
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
    # deprecated
    cal_rand_index,
    cal_adjusted_rand_index,
    cal_cluster_purity,
    cal_nmi,
    cal_chs,
    cal_dbs,
    cal_silhouette,
    cal_internal_cluster_validation_metrics,
    cal_external_cluster_validation_metrics,
)
from .error import (
    calc_mae,
    calc_mse,
    calc_rmse,
    calc_mre,
    calc_quantile_crps,
    calc_quantile_crps_sum,
    # deprecated
    cal_mae,
    cal_mse,
    cal_rmse,
    cal_mre,
)

__all__ = [
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
    # deprecated
    "cal_mae",
    "cal_mse",
    "cal_rmse",
    "cal_mre",
    "cal_rand_index",
    "cal_adjusted_rand_index",
    "cal_cluster_purity",
    "cal_nmi",
    "cal_chs",
    "cal_dbs",
    "cal_silhouette",
    "cal_internal_cluster_validation_metrics",
    "cal_external_cluster_validation_metrics",
    "cal_binary_classification_metrics",
    "cal_precision_recall_f1",
    "cal_pr_auc",
    "cal_roc_auc",
    "cal_acc",
]
