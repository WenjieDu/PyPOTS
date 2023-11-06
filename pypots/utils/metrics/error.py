"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch


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
    assert isinstance(predictions, type(targets)), (
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

    assert isinstance(predictions, type(targets)), (
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
    assert isinstance(predictions, type(targets)), (
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
    assert isinstance(predictions, type(targets)), (
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
