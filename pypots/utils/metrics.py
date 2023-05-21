"""
Utilities for evaluation metrics
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Union, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics


def cal_mae(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Absolute Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import cal_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = cal_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = cal_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.abs(predictions - targets))


def cal_mse(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import cal_mse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mse = cal_mse(predictions, targets)

    mse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`, so the result is 5/5=1.

    If we want to prevent some values from MSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mse = cal_mse(predictions, targets, masks)

    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """

    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.square(predictions - targets))


def cal_rmse(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Root Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import cal_rmse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> rmse = cal_rmse(predictions, targets)

    rmse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`,
    so the result is :math:`\\sqrt{5/5}=1`.

    If we want to prevent some values from RMSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> rmse = cal_rmse(predictions, targets, masks)

    rmse = 0.707 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    return lib.sqrt(cal_mse(predictions, targets, masks))


def cal_mre(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Relative Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import cal_mre
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mre = cal_mre(predictions, targets)

    mre = 0.2 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`,
    so the result is :math:`\\sqrt{3/(1+2+3+4+5)}=1`.

    If we want to prevent some values from MRE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mre = cal_mre(predictions, targets, masks)

    mre = 0.111 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (
            lib.sum(lib.abs(targets * masks)) + 1e-12
        )
    else:
        return lib.sum(lib.abs(predictions - targets)) / (
            lib.sum(lib.abs(targets)) + 1e-12
        )


def cal_binary_classification_metrics(
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

    if len(prob_predictions.shape) == 1 or (
        len(prob_predictions.shape) == 2 and prob_predictions.shape[1] == 1
    ):
        prob_predictions = np.asarray(
            prob_predictions
        ).flatten()  # turn the array shape into [n_samples]
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
    acc_score = cal_acc(prediction_categories, targets)

    # turn targets into binary targets
    mask_val = -1 if pos_label == 0 else 0
    mask = targets == pos_label
    binary_targets = np.copy(targets)
    binary_targets[~mask] = mask_val

    precision, recall, f1 = cal_precision_recall_f1(
        binary_prediction_categories, binary_targets, pos_label
    )
    pr_auc, precisions, recalls, _ = cal_pr_auc(
        binary_predictions, binary_targets, pos_label
    )
    ROC_AUC, fprs, tprs, _ = cal_roc_auc(binary_predictions, binary_targets, pos_label)
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


def cal_precision_recall_f1(
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
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        targets, prob_predictions, pos_label=pos_label
    )
    precision, recall, f1 = precision[pos_label], recall[pos_label], f1[pos_label]
    return precision, recall, f1


def cal_pr_auc(
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

    precisions, recalls, thresholds = metrics.precision_recall_curve(
        targets, prob_predictions, pos_label=pos_label
    )
    pr_auc = metrics.auc(recalls, precisions)
    return pr_auc, precisions, recalls, thresholds


def cal_roc_auc(
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
    fprs, tprs, thresholds = metrics.roc_curve(
        y_true=targets, y_score=prob_predictions, pos_label=pos_label
    )
    roc_auc = metrics.auc(fprs, tprs)
    return roc_auc, fprs, tprs, thresholds


def cal_acc(class_predictions: np.ndarray, targets: np.ndarray) -> float:
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


def cal_rand_index(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate Rand Index, a measure of the similarity between two data clusterings.
    Refer to :cite:`rand1971RandIndex`.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    RI :
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


def cal_adjusted_rand_index(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate adjusted Rand Index. Refer to :cite:`hubert1985AdjustedRI`.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    aRI :
        Adjusted Rand index.

    """
    aRI = metrics.adjusted_rand_score(targets, class_predictions)
    return aRI


def cal_cluster_purity(
    class_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate cluster purity.

    Parameters
    ----------
    class_predictions :
        Clustering results returned by a clusterer.

    targets :
        Ground truth (correct) clustering results.

    Returns
    -------
    cluster_purity :
        cluster purity.

    Notes
    -----
    This function is from the answer https://stackoverflow.com/a/51672699 on StackOverflow.

    """
    contingency_matrix = metrics.cluster.contingency_matrix(targets, class_predictions)
    cluster_purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
        contingency_matrix
    )
    return cluster_purity
