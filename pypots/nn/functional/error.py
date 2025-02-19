"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch


def _check_inputs(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    check_shape: bool = True,
):
    # check type
    assert isinstance(predictions, type(targets)), (
        f"types of `predictions` and `targets` must match, but got"
        f"`predictions`: {type(predictions)}, `target`: {type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    # check shape
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert (
            prediction_shape == target_shape
        ), f"shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}"
    # check NaN
    assert not lib.isnan(predictions).any(), "`predictions` mustn't contain NaN values, but detected NaN in it"
    assert not lib.isnan(targets).any(), "`targets` mustn't contain NaN values, but detected NaN in it"

    if masks is not None:
        # check type
        assert isinstance(masks, type(targets)), (
            f"types of `masks`, `predictions`, and `targets` must match, but got"
            f"`masks`: {type(masks)}, `targets`: {type(targets)}"
        )
        # check shape, masks shape must match targets
        mask_shape = masks.shape
        assert mask_shape == target_shape, (
            f"shape of `masks` must match `targets` shape, "
            f"but got `mask`: {mask_shape} that is different from `targets`: {target_shape}"
        )
        # check NaN
        assert not lib.isnan(masks).any(), "`masks` mustn't contain NaN values, but detected NaN in it"

    return lib


def calc_mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    >>> from pypots.nn.functional import calc_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = calc_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = calc_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (lib.sum(masks) + 1e-12)
    else:
        return lib.mean(lib.abs(predictions - targets))


def calc_mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    >>> from pypots.nn.functional import calc_mse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mse = calc_mse(predictions, targets)

    mse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`, so the result is 5/5=1.

    If we want to prevent some values from MSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mse = calc_mse(predictions, targets, masks)

    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (lib.sum(masks) + 1e-12)
    else:
        return lib.mean(lib.square(predictions - targets))


def calc_rmse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    >>> from pypots.nn.functional import calc_rmse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> rmse = calc_rmse(predictions, targets)

    rmse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`,
    so the result is :math:`\\sqrt{5/5}=1`.

    If we want to prevent some values from RMSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> rmse = calc_rmse(predictions, targets, masks)

    rmse = 0.707 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # don't have to check types and NaN here, since calc_mse() will do it
    lib = np if isinstance(predictions, np.ndarray) else torch
    return lib.sqrt(calc_mse(predictions, targets, masks))


def calc_mre(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    >>> from pypots.nn.functional import calc_mre
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mre = calc_mre(predictions, targets)

    mre = 0.2 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`,
    so the result is :math:`\\sqrt{3/(1+2+3+4+5)}=1`.

    If we want to prevent some values from MRE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mre = calc_mre(predictions, targets, masks)

    mre = 0.111 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (lib.sum(lib.abs(targets * masks)) + 1e-12)
    else:
        return lib.sum(lib.abs(predictions - targets)) / (lib.sum(lib.abs(targets)) + 1e-12)


def calc_quantile_loss(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    q: float,
    eval_points: Union[np.ndarray, torch.Tensor],
) -> Union[float, torch.Tensor]:
    quantile_loss = 2 * torch.sum(
        torch.abs((predictions - targets) * eval_points * ((targets <= predictions) * 1.0 - q))
    )
    return quantile_loss


def calc_quantile_crps(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Union[np.ndarray, torch.Tensor],
    scaler_mean=0,
    scaler_stddev=1,
) -> float:
    """Continuous rank probability score for distributional predictions.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        Only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    scaler_mean:
        Mean value of the scaler used to scale the data.

    scaler_stddev:
        Standard deviation value of the scaler used to scale the data.

    Returns
    -------
    CRPS :
        Value of continuous rank probability score.

    """
    # check shapes and values of inputs
    _ = _check_inputs(predictions, targets, masks, check_shape=False)

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    targets = targets * scaler_stddev + scaler_mean
    predictions = predictions * scaler_stddev + scaler_mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denominator = torch.sum(torch.abs(targets * masks))
    CRPS = torch.tensor(0.0)
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(predictions)):
            q_pred.append(torch.quantile(predictions[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = calc_quantile_loss(q_pred, targets, quantiles[i], masks)
        CRPS += q_loss / denominator
    return CRPS.item() / len(quantiles)


def calc_quantile_crps_sum(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Union[np.ndarray, torch.Tensor],
    scaler_mean=0,
    scaler_stddev=1,
) -> float:
    """Sum continuous rank probability score for distributional predictions.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        Only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    scaler_mean:
        Mean value of the scaler used to scale the data.

    scaler_stddev:
        Standard deviation value of the scaler used to scale the data.

    Returns
    -------
    CRPS :
        Sum value of continuous rank probability score.

    """
    # check shapes and values of inputs
    _ = _check_inputs(predictions, targets, masks, check_shape=False)

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    masks = masks.mean(-1)
    targets = targets * scaler_stddev + scaler_mean
    targets = targets.sum(-1)
    predictions = predictions * scaler_stddev + scaler_mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denominator = torch.sum(torch.abs(targets * masks))
    CRPS = torch.tensor(0.0)
    for i in range(len(quantiles)):
        q_pred = torch.quantile(predictions.sum(-1), quantiles[i], dim=1)
        q_loss = calc_quantile_loss(q_pred, targets, quantiles[i], masks)
        CRPS += q_loss / denominator
    return CRPS.item() / len(quantiles)
