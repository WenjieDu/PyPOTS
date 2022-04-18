"""
Utilities for evaluation metrics
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np
import torch
from sklearn import metrics


def cal_mae(inputs, target, mask=None):
    """ calculate Mean Absolute Error"""
    assert type(inputs) == type(target), f'types of inputs and target must match, ' \
                                         f'type(inputs)={type(inputs)}, type(target)={type(target)}'
    lib = np if isinstance(inputs, np.ndarray) else torch
    if mask is not None:
        return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.abs(inputs - target))


def cal_mse(inputs, target, mask=None):
    """ calculate Mean Square Error"""
    assert type(inputs) == type(target), f'types of inputs and target must match, ' \
                                         f'type(inputs)={type(inputs)}, type(target)={type(target)}'
    lib = np if isinstance(inputs, np.ndarray) else torch
    if mask is not None:
        return lib.sum(lib.square(inputs - target) * mask) / (lib.sum(mask) + 1e-9)
    else:
        return lib.mean(lib.square(inputs - target))


def cal_rmse(inputs, target, mask=None):
    """ calculate Root Mean Square Error"""
    assert type(inputs) == type(target), f'types of inputs and target must match, ' \
                                         f'type(inputs)={type(inputs)}, type(target)={type(target)}'
    lib = np if isinstance(inputs, np.ndarray) else torch
    return lib.sqrt(cal_mse(inputs, target, mask))


def cal_mre(inputs, target, mask=None):
    """ calculate Mean Relative Error"""
    assert type(inputs) == type(target), f'types of inputs and target must match, ' \
                                         f'type(inputs)={type(inputs)}, type(target)={type(target)}'
    lib = np if isinstance(inputs, np.ndarray) else torch
    if mask is not None:
        return lib.sum(lib.abs(inputs - target) * mask) / (lib.sum(lib.abs(target * mask)) + 1e-9)
    else:
        return lib.mean(lib.abs(inputs - target)) / (lib.sum(lib.abs(target)) + 1e-9)


def cal_binary_classification_metrics(predictions, targets, pos_label=1):
    """ Calculate the evaluation metrics for the binary classification task,
        including accuracy, precision, recall, f1 score, area under ROC curve, and area under Precision-Recall curve

    Parameters
    ----------
    predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated predictions (probabilities) returned by a classifier or a decision function.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.
    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    classification_metrics : dict
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
        raise f'targets dimensions should be 1 or 2, but got targets.shape: {targets.shape}'

    if len(predictions.shape) == 1 or (len(predictions.shape) == 2 and predictions.shape[1] == 1):
        predictions = np.asarray(predictions).flatten()  # turn the array shape into [n_samples]
        binary_predictions = predictions
        prediction_categories = (predictions >= 0.5).astype(int)
    elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
        prediction_categories = np.argmax(predictions, axis=1)
        binary_predictions = np.take(predictions, prediction_categories)
    else:
        raise f'predictions dimensions should be 1 or 2, but got predictions.shape: {predictions.shape}'

    precision, recall, f1 = cal_precision_recall_f1(prediction_categories, targets, pos_label=pos_label)
    pr_auc, precisions, recalls, _ = cal_pr_auc(binary_predictions, targets, pos_label=pos_label)
    acc_score = cal_acc(prediction_categories, targets)
    ROC_AUC, fprs, tprs, _ = cal_roc_auc(binary_predictions, targets)
    PR_AUC = metrics.auc(recalls, precisions)
    classification_metrics = {
        'predictions': prediction_categories,
        'accuracy': acc_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precisions': precisions,
        'recalls': recalls,
        'pr_auc': PR_AUC,
        'fprs': fprs,
        'tprs': tprs,
        'roc_auc': ROC_AUC,
    }
    return classification_metrics


def cal_precision_recall_f1(predictions, targets, pos_label=1):
    """ Calculate precision, recall, and F1-score of model predictions.

    Parameters
    ----------
    predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated predictions (probabilities) returned by a classifier or a decision function.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.
    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    precision : float
        The precision value of model predictions.
    recall : float
        The recall value of model predictions.
    f1 : float
        The F1 score of model predictions.

    """
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(targets, predictions,
                                                                       pos_label=pos_label)
    precision, recall, f1 = precision[1], recall[1], f1[1]
    return precision, recall, f1


def cal_pr_auc(predictions, targets, pos_label=1):
    """ Calculate precisions, recalls, and area under PR curve of model predictions.

    Parameters
    ----------
    predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated predictions (probabilities) returned by a classifier or a decision function.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.
    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    pr_auc : float
        Value of area under Precision-Recall curve.
    precisions : array-like
        Precision values of Precision-Recall curve.
    recalls : array-like
        Recall values of Precision-Recall curve.
    thresholds : array-like
        Increasing thresholds on the decision function used to compute precision and recall.

    """

    precisions, recalls, thresholds = metrics.precision_recall_curve(targets, predictions,
                                                                     pos_label=pos_label)
    pr_auc = metrics.auc(recalls, precisions)
    return pr_auc, precisions, recalls, thresholds


def cal_roc_auc(predictions, targets, pos_label=1):
    """ Calculate false positive rates, true positive rates, and area under AUC curve of model predictions.

    Parameters
    ----------
    predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated predictions (probabilities) returned by a classifier or a decision function.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.
    pos_label: int, default=1
        The label of the positive class.

    Returns
    -------
    roc_auc : float
        The area under ROC curve.
    fprs : array-like
        False positive rates of ROC curve.
    tprs : array-like
        True positive rates of ROC curve.
    thresholds : array-like
        Increasing thresholds on the decision function used to compute FPR and TPR.

    """
    fprs, tprs, thresholds = metrics.roc_curve(y_true=targets, y_score=predictions,
                                               pos_label=pos_label)
    roc_auc = metrics.auc(fprs, tprs)
    return roc_auc, fprs, tprs, thresholds


def cal_acc(predictions, targets):
    """ Calculate accuracy score of model predictions.

    Parameters
    ----------
    predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated predictions (probabilities) returned by a classifier or a decision function.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.

    Returns
    -------
    acc_score : float
        The accuracy of model predictions.

    """
    acc_score = metrics.accuracy_score(targets, predictions)
    return acc_score
