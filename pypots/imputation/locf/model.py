"""
The implementation of LOCF (Last Observed Carried Forward) for the partially-observed time-series imputation task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import warnings
from typing import Union, Optional

import numpy as np
import torch

from ..base import BaseImputer
from ...utils.logging import logger


class LOCF(BaseImputer):
    """LOCF (Last Observed Carried Forward) imputation method. A naive imputation method that fills missing values
    with the last observed value. Simple but commonly used in practice.

    Parameters
    ----------
    first_step_imputation : str, default='backward'
        With LOCF, the observed values are carried forward to impute the missing ones. But if the first value
        is missing, there is no value to carry forward. This parameter is used to determine the strategy to
        impute the missing values at the beginning of the time-series sequence after LOCF is applied.
        It can be one of ['backward', 'zero', 'mean', 'nan'].
        If 'nan', the missing values at the sequence beginning will be left as NaNs.
        If 'zero', the missing values at the sequence beginning will be imputed with 0.
        If 'backward', the missing values at the beginning of the time-series sequence will be imputed with the
        first observed value in the sequence, i.e. the first observed value will be carried backward to impute
        the missing values at the beginning of the sequence.
        If 'mean', the missing values at the sequence beginning will be imputed with the overall mean
        values of features in the dataset.

    """

    def __init__(
        self,
        first_step_imputation: str = "backward",
        device: Optional[Union[str, torch.device, list]] = None,
    ):
        super().__init__(device=device)
        assert first_step_imputation in ["nan", "zero", "backward", "mean"]
        self.first_step_imputation = first_step_imputation

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "h5py",
    ) -> None:
        """Train the imputer on the given data.

        Warnings
        --------
        LOCF does not need to run fit().
        Please run func ``predict()`` directly.

        """
        warnings.warn(
            "LOCF (Last Observed Carried Forward) imputation class has no parameter to train. "
            "Please run func impute(X) directly."
        )

    def _locf_numpy(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Numpy implementation of LOCF.

        Parameters
        ----------
        X : np.ndarray,
            Time series containing missing values (NaN) to be imputed.

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
            if self.first_step_imputation == "nan":
                pass
            elif self.first_step_imputation == "zero":
                X_imputed = np.nan_to_num(X_imputed, nan=0)
            elif self.first_step_imputation == "backward":
                X_imputed_transpose = np.copy(X_imputed)
                # imputed by last observation carried backward (LOCB)
                X_imputed_transpose = np.flip(X_imputed_transpose, axis=1)
                X_LOCB = self._locf_numpy(X_imputed_transpose)
                assert not np.isnan(X_LOCB).any(), "NaN still exists in X_LOCB"
                X_imputed = np.flip(X_LOCB, axis=1)
            elif self.first_step_imputation == "mean":
                bz, n_steps, n_features = X_imputed.shape
                X_imputed_squeezed = np.copy(X_imputed).reshape(-1, n_features)
                mean_values = np.nanmean(X_imputed_squeezed, axis=0)
                for i, v in enumerate(mean_values):
                    X_imputed[:, :, i] = np.nan_to_num(X_imputed[:, :, i], nan=v)

        return X_imputed

    def _locf_torch(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Torch implementation of LOCF.

        Parameters
        ----------
        X : tensor,
            Time series containing missing values (NaN) to be imputed.

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
            if self.first_step_imputation == "nan":
                pass
            elif self.first_step_imputation == "zero":
                X_imputed = torch.nan_to_num(X_imputed, nan=0)
            elif self.first_step_imputation == "backward":
                X_imputed_transpose = X_imputed.clone()
                # imputed by last observation carried backward (LOCB)
                X_imputed_transpose = torch.flip(X_imputed_transpose, dims=[1])
                X_LOCB = self._locf_torch(X_imputed_transpose)
                assert not torch.isnan(X_LOCB).any(), "NaN still exists in X_LOCB"
                X_imputed = torch.flip(X_LOCB, dims=[1])
            elif self.first_step_imputation == "mean":
                bz, n_steps, n_features = X_imputed.shape
                X_imputed_squeezed = X_imputed.clone().reshape(-1, n_features)
                mean_values = torch.nanmean(X_imputed_squeezed, dim=0)
                for i, v in enumerate(mean_values):
                    X_imputed[:, :, i] = torch.nan_to_num(X_imputed[:, :, i], nan=v)

        return X_imputed

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "h5py",
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set : dict or str
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type : str
            The type of the given file if test_set is a path string.

        Returns
        -------
        result_dict: dict
            Prediction results in a Python Dictionary for the given samples.
            It should be a dictionary including keys as 'imputation', 'classification', 'clustering', and 'forecasting'.
            For sure, only the keys that relevant tasks are supported by the model will be returned.
        """
        assert not isinstance(test_set, str)
        X = test_set["X"]

        assert len(X.shape) == 3, (
            f"Input X should have 3 dimensions [n_samples, n_steps, n_features], "
            f"but the actual shape of X: {X.shape}"
        )
        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(X, np.ndarray):
            imputed_data = self._locf_numpy(X)
        elif isinstance(X, torch.Tensor):
            imputed_data = self._locf_torch(X)
        else:
            raise TypeError(
                "X must be type of list/np.ndarray/torch.Tensor, " f"but got {type(X)}"
            )

        result_dict = {
            "imputation": imputed_data,
        }
        return result_dict

    def impute(
        self,
        X: Union[dict, str],
        file_type="h5py",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Warnings
        --------
        The method impute is deprecated. Please use `predict()` instead.

        Parameters
        ----------
        X :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        logger.warning(
            "🚨DeprecationWarning: The method impute is deprecated. Please use `predict` instead."
        )
        results_dict = self.predict(X, file_type=file_type)
        return results_dict["imputation"]
