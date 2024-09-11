"""
The implementation of Mean value imputation.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import warnings
from typing import Union, Optional

import h5py
import numpy as np
import torch

from ..base import BaseImputer


class Mean(BaseImputer):
    """Mean value imputation method."""

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
        Mean imputation class does not need to run fit().
        Please run func ``predict()`` directly.

        """
        warnings.warn("Mean imputation class has no parameter to train. Please run func `predict()` directly.")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set : dict or str
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        result_dict: dict
            Prediction results in a Python Dictionary for the given samples.
            It should be a dictionary including keys as 'imputation', 'classification', 'clustering', and 'forecasting'.
            For sure, only the keys that relevant tasks are supported by the model will be returned.
        """
        if isinstance(test_set, str):
            with h5py.File(test_set, "r") as f:
                X = f["X"][:]
        else:
            X = test_set["X"]

        assert len(X.shape) == 3, (
            f"Input X should have 3 dimensions [n_samples, n_steps, n_features], "
            f"but the actual shape of X: {X.shape}"
        )
        if isinstance(X, list):
            X = np.asarray(X)

        n_samples, n_steps, n_features = X.shape

        if isinstance(X, np.ndarray):
            X_imputed_reshaped = np.copy(X).reshape(-1, n_features)
            mean_values = np.nanmean(X_imputed_reshaped, axis=0)
            for i, v in enumerate(mean_values):
                X_imputed_reshaped[:, i] = np.nan_to_num(X_imputed_reshaped[:, i], nan=v)
            imputed_data = X_imputed_reshaped.reshape(n_samples, n_steps, n_features)
        elif isinstance(X, torch.Tensor):
            X_imputed_reshaped = torch.clone(X).reshape(-1, n_features)
            mean_values = torch.nanmean(X_imputed_reshaped, dim=0).numpy()
            for i, v in enumerate(mean_values):
                X_imputed_reshaped[:, i] = torch.nan_to_num(X_imputed_reshaped[:, i], nan=v)
            imputed_data = X_imputed_reshaped.reshape(n_samples, n_steps, n_features)
        else:
            raise ValueError()

        result_dict = {
            "imputation": imputed_data,
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]
