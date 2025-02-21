"""
Evaluation metrics related to classification.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from ..logging import logger
from ...nn.functional.classification import (
    calc_binary_classification_metrics,
    calc_precision_recall_f1,
    calc_pr_auc,
    calc_roc_auc,
    calc_acc,
)

# pypots.nn.functional.classification is deprecated, and moved to pypots.nn.functional.classification
logger.warning(
    "🚨 Please import from pypots.nn.functional.classification instead of pypots.nn.functional.classification"
)

__all__ = [
    "calc_binary_classification_metrics",
    "calc_precision_recall_f1",
    "calc_pr_auc",
    "calc_roc_auc",
    "calc_acc",
]
