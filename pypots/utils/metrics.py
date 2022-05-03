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


def cal_binary_classification_metrics(prob_predictions, targets, pos_label=1):
    """ Calculate the evaluation metrics for the binary classification task,
        including accuracy, precision, recall, f1 score, area under ROC curve, and area under Precision-Recall curve.
        If targets contains multiple categories, please set the positive category as `pos_label`.

    Parameters
    ----------
    prob_predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated probability predictions returned by a decision function.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.
    pos_label : int, default=1
        The label of the positive class.
        Note that pos_label is also the index used to extract binary prediction probabilities from `predictions`.

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
        raise f'predictions dimensions should be 1 or 2, but got predictions.shape: {prob_predictions.shape}'

    # accuracy score doesn't have to be of binary classification
    acc_score = cal_acc(prediction_categories, targets)

    # turn targets into binary targets
    mask_val = -1 if pos_label == 0 else 0
    mask = targets == pos_label
    binary_targets = np.copy(targets)
    binary_targets[~mask] = mask_val

    precision, recall, f1 = cal_precision_recall_f1(binary_prediction_categories, binary_targets, pos_label)
    pr_auc, precisions, recalls, _ = cal_pr_auc(binary_predictions, binary_targets, pos_label)
    ROC_AUC, fprs, tprs, _ = cal_roc_auc(binary_predictions, binary_targets, pos_label)
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


def cal_precision_recall_f1(prob_predictions, targets, pos_label=1):
    """ Calculate precision, recall, and F1-score of model predictions.

    Parameters
    ----------
    prob_predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated probability predictions returned by a decision function.
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
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(targets, prob_predictions,
                                                                       pos_label=pos_label)
    precision, recall, f1 = precision[pos_label], recall[pos_label], f1[pos_label]
    return precision, recall, f1


def cal_pr_auc(prob_predictions, targets, pos_label=1):
    """ Calculate precisions, recalls, and area under PR curve of model predictions.

    Parameters
    ----------
    prob_predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated probability predictions returned by a decision function.
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

    precisions, recalls, thresholds = metrics.precision_recall_curve(targets, prob_predictions,
                                                                     pos_label=pos_label)
    pr_auc = metrics.auc(recalls, precisions)
    return pr_auc, precisions, recalls, thresholds


def cal_roc_auc(prob_predictions, targets, pos_label=1):
    """ Calculate false positive rates, true positive rates, and area under AUC curve of model predictions.

    Parameters
    ----------
    prob_predictions : array-like, 1d, [n_samples]
        Estimated probabilities/predictions returned by a decision function.
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
    fprs, tprs, thresholds = metrics.roc_curve(y_true=targets, y_score=prob_predictions,
                                               pos_label=pos_label)
    roc_auc = metrics.auc(fprs, tprs)
    return roc_auc, fprs, tprs, thresholds


def cal_acc(class_predictions, targets):
    """ Calculate accuracy score of model predictions.

    Parameters
    ----------
    class_predictions : array-like, 1d or 2d, [n_samples] or [n_samples, n_categories]
        Estimated classification predictions returned by a classifier.
    targets : array-like, 1d or 2d, shape of [n_samples] or [n_samples, 1]
        Ground truth (correct) classification results.

    Returns
    -------
    acc_score : float
        The accuracy of model predictions.

    """
    acc_score = metrics.accuracy_score(targets, class_predictions)
    return acc_score


def cal_rand_index(class_predictions, targets):
    """ Calculate Rand Index, a measure of the similarity between two data clusterings.
        Refer to :cite:`rand1971RandIndex`.

    Parameters
    ----------
    class_predictions : array
        Clustering results returned by a clusterer.
    targets : array
        Ground truth (correct) clustering results.

    Returns
    -------
    RI : float
        Rand index.
    """
    # # detailed implementation
    # n = len(targets)
    # TP = 0
    # TN = 0
    # for i in range(n - 1):
    #     for j in range(i + 1, n):
    #         if targets[i] != targets[j]:
    #             if class_predictions[i] != class_predictions[j]:
    #                 TN += 1
    #         else:
    #             if class_predictions[i] == class_predictions[j]:
    #                 TP += 1
    #
    # RI = n * (n - 1) / 2
    # RI = (TP + TN) / RI

    RI = metrics.rand_score(targets, class_predictions)

    return RI


def cal_adjusted_rand_index(class_predictions, targets):
    """ Calculate adjusted Rand Index.
    Refer to :cite:`hubert1985AdjustedRI`.

    Parameters
    ----------
    class_predictions : array
        Clustering results returned by a clusterer.
    targets : array
        Ground truth (correct) clustering results.

    Returns
    -------
    aRI : float
        Adjusted Rand index.
    """
    aRI = metrics.adjusted_rand_score(targets, class_predictions)
    return aRI


def cal_cluster_purity(class_predictions, targets):
    """ Calculate cluster purity.

    Parameters
    ----------
    class_predictions : array
        Clustering results returned by a clusterer.
    targets : array
        Ground truth (correct) clustering results.

    Returns
    -------
    cluster_purity : float
        cluster purity.

    Notes
    -----
    This function is from the answer https://stackoverflow.com/a/51672699 on StackOverflow.

    """
    contingency_matrix = metrics.cluster.contingency_matrix(targets, class_predictions)
    cluster_purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return cluster_purity
