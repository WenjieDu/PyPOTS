"""
Evaluation metrics related to clustering.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

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
from ..logging import logger

# pypots.nn.functional.clustering is deprecated, and moved to pypots.nn.functional.clustering
logger.warning("🚨 Please import from pypots.nn.functional.clustering instead of pypots.nn.functional.clustering")

__all__ = [
    "calc_adjusted_rand_index",
    "calc_chs",
    "calc_cluster_purity",
    "calc_dbs",
    "calc_external_cluster_validation_metrics",
    "calc_internal_cluster_validation_metrics",
    "calc_nmi",
    "calc_rand_index",
    "calc_silhouette",
]
