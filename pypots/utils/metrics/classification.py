"""
Evaluation metrics related to classification.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from ...nn.functional.classification import (
    calc_acc,
    calc_binary_classification_metrics,
    calc_pr_auc,
    calc_precision_recall_f1,
    calc_roc_auc,
)
from ..logging import logger

# pypots.nn.functional.classification is deprecated, and moved to pypots.nn.functional.classification
logger.warning(
    "🚨 Please import from pypots.nn.functional.classification instead of pypots.nn.functional.classification"
)

__all__ = [
    "calc_acc",
    "calc_binary_classification_metrics",
    "calc_pr_auc",
    "calc_precision_recall_f1",
    "calc_roc_auc",
]
