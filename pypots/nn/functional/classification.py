"""
Evaluation metrics related to classification.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Tuple

import numpy as np
from sklearn import metrics


def calc_binary_classification_metrics(
    prob_predictions: np.ndarray,
    targets: np.ndarray,
    pos_label: int = 1,
) -> dict:
    """Calculate the evaluation metrics for the binary classification task,
    including accuracy, precision, recall, f1 score, area under ROC curve, and area under Precision-Recall curve.
    If targets contains multiple categories, please set the positive category as `pos_label`.

    Parameters
    ----------
    prob_predictions :
        Estimated probability predictions returned by a decision function.

    targets :
        Ground truth (correct) classification results.

    pos_label :
        The label of the positive class.
        Note that pos_label is also the index used to extract binary prediction probabilities from `predictions`.

    Returns
    -------
    classification_metrics :
        A dictionary contains classification metrics and useful results:

        predictions: binary categories of the prediction results;

        accuracy: prediction accuracy;

        precision: prediction precision;

        recall: prediction recall;

        f1: F1-score;

        precisions: precision values of Precision-Recall curve

        recalls: recall values of Precision-Recall curve

        pr_auc: area under Precision-Recall curve

        fprs: false positive rates of ROC curve

        tprs: true positive rates of ROC curve

        roc_auc: area under ROC curve

    """
    # check the dimensionality
    if len(targets.shape) == 1:
        pass
    elif len(targets.shape) == 2 and targets.shape[1] == 1:
        targets = np.asarray(targets).flatten()
    else:
        raise f"targets dimensions should be 1 or 2, but got targets.shape: {targets.shape}"

    if len(prob_predictions.shape) == 1 or (len(prob_predictions.shape) == 2 and prob_predictions.shape[1] == 1):
        prob_predictions = np.asarray(prob_predictions).flatten()  # turn the array shape into [n_samples]
        binary_predictions = prob_predictions
        prediction_categories = (prob_predictions >= 0.5).astype(int)
        binary_prediction_categories = prediction_categories
    elif len(prob_predictions.shape) == 2 and prob_predictions.shape[1] > 1:
        prediction_categories = np.argmax(prob_predictions, axis=1)
        binary_predictions = prob_predictions[:, pos_label]
        binary_prediction_categories = (prediction_categories == pos_label).astype(int)
    else:
        raise f"predictions dimensions should be 1 or 2, but got predictions.shape: {prob_predictions.shape}"

    # accuracy score doesn't have to be of binary classification
    acc_score = calc_acc(prediction_categories, targets)

    # turn targets into binary targets
    mask_val = -1 if pos_label == 0 else 0
    mask = targets == pos_label
    binary_targets = np.copy(targets)
    binary_targets[~mask] = mask_val

    precision, recall, f1 = calc_precision_recall_f1(binary_prediction_categories, binary_targets, pos_label)
    pr_auc, precisions, recalls, _ = calc_pr_auc(binary_predictions, binary_targets, pos_label)
    ROC_AUC, fprs, tprs, _ = calc_roc_auc(binary_predictions, binary_targets, pos_label)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        "predictions": prediction_categories,
        "accuracy": acc_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precisions": precisions,
        "recalls": recalls,
        "pr_auc": PR_AUC,
        "fprs": fprs,
        "tprs": tprs,
        "roc_auc": ROC_AUC,
    }
    return classification_metrics


def calc_precision_recall_f1(
    prob_predictions: np.ndarray,
    targets: np.ndarray,
    pos_label: int = 1,
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1-score of model predictions.

    Parameters
    ----------
    prob_predictions :
        Estimated probability predictions returned by a decision function.

    targets :
        Ground truth (correct) classification results.

    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    precision :
        The precision value of model predictions.

    recall :
        The recall value of model predictions.

    f1 :
        The F1 score of model predictions.

    """
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(targets, prob_predictions, pos_label=pos_label)
    precision, recall, f1 = precision[pos_label], recall[pos_label], f1[pos_label]
    return precision, recall, f1


def calc_pr_auc(
    prob_predictions: np.ndarray,
    targets: np.ndarray,
    pos_label: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate precisions, recalls, and area under PR curve of model predictions.

    Parameters
    ----------
    prob_predictions :
        Estimated probability predictions returned by a decision function.

    targets :
        Ground truth (correct) classification results.

    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    pr_auc :
        Value of area under Precision-Recall curve.

    precisions :
        Precision values of Precision-Recall curve.

    recalls :
        Recall values of Precision-Recall curve.

    thresholds :
        Increasing thresholds on the decision function used to compute precision and recall.

    """

    precisions, recalls, thresholds = metrics.precision_recall_curve(targets, prob_predictions, pos_label=pos_label)
    pr_auc = metrics.auc(recalls, precisions)
    return pr_auc, precisions, recalls, thresholds


def calc_roc_auc(
    prob_predictions: np.ndarray,
    targets: np.ndarray,
    pos_label: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate false positive rates, true positive rates, and area under AUC curve of model predictions.

    Parameters
    ----------
    prob_predictions :
        Estimated probabilities/predictions returned by a decision function.

    targets :
        Ground truth (correct) classification results.

    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    roc_auc :
        The area under ROC curve.

    fprs :
        False positive rates of ROC curve.

    tprs :
        True positive rates of ROC curve.

    thresholds :
        Increasing thresholds on the decision function used to compute FPR and TPR.

    """
    fprs, tprs, thresholds = metrics.roc_curve(y_true=targets, y_score=prob_predictions, pos_label=pos_label)
    roc_auc = metrics.auc(fprs, tprs)
    return roc_auc, fprs, tprs, thresholds


def calc_acc(class_predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate accuracy score of model predictions.

    Parameters
    ----------
    class_predictions :
        Estimated classification predictions returned by a classifier.

    targets :
        Ground truth (correct) classification results.

    Returns
    -------
    acc_score :
        The accuracy of model predictions.

    """
    acc_score = metrics.accuracy_score(targets, class_predictions)
    return acc_score
