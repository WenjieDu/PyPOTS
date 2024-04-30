"""
The core wrapper assembles the submodules of LOCF imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch


def locf_numpy(
    X: np.ndarray,
    first_step_imputation: str = "backward",
) -> np.ndarray:
    """Numpy implementation of LOCF.

    Parameters
    ----------
    X : np.ndarray,
        Time series containing missing values (NaN) to be imputed.

    first_step_imputation : str, default='backward'
        With LOCF, the observed values are carried forward to impute the missing ones. But if the first value
        is missing, there is no value to carry forward. This parameter is used to determine the strategy to
        impute the missing values at the beginning of the time-series sequence after LOCF is applied.
        It can be one of ['backward', 'zero', 'median', 'nan'].
        If 'nan', the missing values at the sequence beginning will be left as NaNs.
        If 'zero', the missing values at the sequence beginning will be imputed with 0.
        If 'backward', the missing values at the beginning of the time-series sequence will be imputed with the
        first observed value in the sequence, i.e. the first observed value will be carried backward to impute
        the missing values at the beginning of the sequence. This method is also known as NOCB (Next Observation
        Carried Backward). If 'median', the missing values at the sequence beginning will be imputed with the overall
        median values of features in the dataset.
        If `first_step_imputation` is not "nan", if missing values still exist (this is usually caused by whole feature
        missing) after applying `first_step_imputation`, they will be filled with 0.

    Returns
    -------
    X_imputed : array,
        Imputed time series.

    Notes
    -----
    This implementation gets inspired by the question on StackOverflow:
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array

    """
    trans_X = X.transpose((0, 2, 1))
    mask = np.isnan(trans_X)
    n_samples, n_steps, n_features = mask.shape
    idx = np.where(~mask, np.arange(n_features), 0)
    idx = np.maximum.accumulate(idx, axis=2)

    collector = []
    for x, i in zip(trans_X, idx):
        collector.append(x[np.arange(n_steps)[:, None], i])
    X_imputed = np.asarray(collector)
    X_imputed = X_imputed.transpose((0, 2, 1))

    # If there are values still missing, they are missing at the beginning of the time-series sequence.
    if np.isnan(X_imputed).any():
        if first_step_imputation == "nan":
            pass
        elif first_step_imputation == "zero":
            X_imputed = np.nan_to_num(X_imputed, nan=0)
        elif first_step_imputation == "backward":
            # imputed by last observation carried backward (LOCB)
            X_imputed_transpose = np.copy(X_imputed)
            X_imputed_transpose = np.flip(X_imputed_transpose, axis=1)
            X_LOCB = locf_numpy(
                X_imputed_transpose,
                "zero",
            )
            X_imputed = np.flip(X_LOCB, axis=1)
        elif first_step_imputation == "median":
            bz, n_steps, n_features = X_imputed.shape
            X_imputed_reshaped = np.copy(X_imputed).reshape(-1, n_features)
            median_values = np.nanmedian(X_imputed_reshaped, axis=0)
            for i, v in enumerate(median_values):
                X_imputed[:, :, i] = np.nan_to_num(X_imputed[:, :, i], nan=v)
            if np.isnan(X_imputed).any():
                X_imputed = np.nan_to_num(X_imputed, nan=0)

    return X_imputed


def locf_torch(
    X: torch.Tensor,
    first_step_imputation: str = "backward",
) -> torch.Tensor:
    """Torch implementation of LOCF.

    Parameters
    ----------
    X : tensor,
        Time series containing missing values (NaN) to be imputed.

    first_step_imputation : str, default='backward'
        With LOCF, the observed values are carried forward to impute the missing ones. But if the first value
        is missing, there is no value to carry forward. This parameter is used to determine the strategy to
        impute the missing values at the beginning of the time-series sequence after LOCF is applied.
        It can be one of ['backward', 'zero', 'median', 'nan'].
        If 'nan', the missing values at the sequence beginning will be left as NaNs.
        If 'zero', the missing values at the sequence beginning will be imputed with 0.
        If 'backward', the missing values at the beginning of the time-series sequence will be imputed with the
        first observed value in the sequence, i.e. the first observed value will be carried backward to impute
        the missing values at the beginning of the sequence. This method is also known as NOCB (Next Observation
        Carried Backward). If 'median', the missing values at the sequence beginning will be imputed with the overall
        median values of features in the dataset.
        If `first_step_imputation` is not "nan", if missing values still exist (this is usually caused by whole feature
        missing) after applying `first_step_imputation`, they will be filled with 0.

    Returns
    -------
    X_imputed : tensor,
        Imputed time series.
    """
    trans_X = X.permute((0, 2, 1))
    mask = torch.isnan(trans_X)
    n_samples, n_steps, n_features = mask.shape
    idx = torch.where(~mask, torch.arange(n_features, device=mask.device), 0)
    idx = np.maximum.accumulate(idx, axis=2)

    collector = []
    for x, i in zip(trans_X, idx):
        collector.append(x[torch.arange(n_steps)[:, None], i])
    X_imputed = torch.stack(collector)
    X_imputed = X_imputed.permute((0, 2, 1))

    # If there are values still missing, they are missing at the beginning of the time-series sequence.
    if torch.isnan(X_imputed).any():
        if first_step_imputation == "nan":
            pass
        elif first_step_imputation == "zero":
            X_imputed = torch.nan_to_num(X_imputed, nan=0)
        elif first_step_imputation == "backward":
            # imputed by last observation carried backward (LOCB)
            X_imputed_transpose = X_imputed.clone()
            X_imputed_transpose = torch.flip(X_imputed_transpose, dims=[1])
            X_LOCB = locf_torch(
                X_imputed_transpose,
                "zero",
            )
            X_imputed = torch.flip(X_LOCB, dims=[1])
        elif first_step_imputation == "median":
            bz, n_steps, n_features = X_imputed.shape
            X_imputed_reshaped = X_imputed.clone().reshape(-1, n_features)
            median_values = torch.nanmedian(X_imputed_reshaped, dim=0)
            for i, v in enumerate(median_values.values):
                X_imputed[:, :, i] = torch.nan_to_num(X_imputed[:, :, i], nan=v)
            if torch.isnan(X_imputed).any():
                X_imputed = torch.nan_to_num(X_imputed, nan=0)

    return X_imputed
