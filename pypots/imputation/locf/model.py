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
    """LOCF (Last Observed Carried Forward) imputation method.
    A naive imputation method that fills missing values with the last observed value.
    Simple but commonly used in practice.

    Parameters
    ----------
    nan : float, default=0,
        The value used to impute data missing at the beginning of the sequence.
    """

    def __init__(self, nan: Optional[Union[float, int]] = 0):
        super().__init__("cpu")
        self.nan = nan

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
        Please run func ``impute()`` directly.

        """
        warnings.warn(
            "LOCF (Last Observed Carried Forward) imputation class has no parameter to train. "
            "Please run func impute(X) directly."
        )

    def _locf_numpy(self, X: np.ndarray) -> np.ndarray:
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
        np.maximum.accumulate(idx, axis=2, out=idx)

        collector = []
        for x, i in zip(trans_X, idx):
            collector.append(x[np.arange(n_steps)[:, None], i])
        X_imputed = np.asarray(collector)
        X_imputed = X_imputed.transpose((0, 2, 1))

        # If there are values still missing,
        # they are missing at the beginning of the time-series sequence.
        # Impute them with self.nan
        if np.isnan(X_imputed).any():
            X_imputed = np.nan_to_num(X_imputed, nan=self.nan)

        return X_imputed

    def _locf_torch(self, X: torch.Tensor) -> torch.Tensor:
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
        idx = torch.cummax(idx, dim=2)

        collector = []
        for x, i in zip(trans_X, idx):
            collector.append(x[torch.arange(n_steps)[:, None], i])
        X_imputed = torch.concat(collector, dim=0)
        X_imputed = X_imputed.permute((0, 2, 1))

        # If there are values still missing,
        # they are missing at the beginning of the time-series sequence.
        # Impute them with self.nan
        if torch.isnan(X_imputed).any():
            X_imputed = torch.nan_to_num(X_imputed, nan=self.nan)

        return X_imputed

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "h5py",
    ) -> dict:
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
            imputed_data = self._locf_torch(X).detach().cpu().numpy()
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
        logger.warning(
            "🚨DeprecationWarning: The method impute is deprecated. Please use `predict` instead."
        )
        results_dict = self.predict(X, file_type=file_type)
        return results_dict["imputation"]
