"""
The implementation of Median value imputation.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import warnings
from typing import Union, Optional

import h5py
import numpy as np
import torch

from ..base import BaseImputer


class Median(BaseImputer):
    """Median value imputation method."""

    def __init__(
        self,
    ):
        super().__init__()

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the imputer on the given data.

        Warnings
        --------
        Median imputation class does not need to run fit().
        Please run func ``predict()`` directly.

        """
        warnings.warn("Median imputation class has no parameter to train. Please run func `predict()` directly.")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:

        if isinstance(test_set, str):
            with h5py.File(test_set, "r") as f:
                X = f["X"][:]
        else:
            X = test_set["X"]

        if isinstance(X, list):
            X = np.asarray(X)

        assert len(X.shape) == 3, (
            f"Input X should have 3 dimensions [n_samples, n_steps, n_features], "
            f"but the actual shape of X: {X.shape}"
        )

        n_samples, n_steps, n_features = X.shape

        if isinstance(X, np.ndarray):
            X_imputed_reshaped = np.copy(X).reshape(-1, n_features)
            median_values = np.nanmedian(X_imputed_reshaped, axis=0)
            # Use 0.0 as fallback for features where all values are NaN
            median_values = np.nan_to_num(median_values, nan=0.0)
            for i, v in enumerate(median_values):
                X_imputed_reshaped[:, i] = np.nan_to_num(X_imputed_reshaped[:, i], nan=v)
            imputed_data = X_imputed_reshaped.reshape(n_samples, n_steps, n_features)
        elif isinstance(X, torch.Tensor):
            X_imputed_reshaped = torch.clone(X).reshape(-1, n_features)
            median_values = torch.nanmedian(X_imputed_reshaped, dim=0).values
            # Use 0.0 as fallback for features where all values are NaN
            median_values = torch.nan_to_num(median_values, nan=0.0)
            nan_mask = torch.isnan(X_imputed_reshaped)
            X_imputed_reshaped = torch.where(
                nan_mask, median_values.unsqueeze(0).expand_as(X_imputed_reshaped), X_imputed_reshaped
            )
            imputed_data = X_imputed_reshaped.reshape(n_samples, n_steps, n_features)
        else:
            raise ValueError(
                f"Input X must be numpy.ndarray or torch.Tensor, but got {type(X)}"
            )

        result_dict = {
            "imputation": imputed_data,
        }
        return result_dict
