"""
Implementation of the imputation method LOCF (Last Observed Carried Forward).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import warnings

import numpy as np
import torch

from pypots.imputation.base import BaseImputer


class LOCF(BaseImputer):
    """ LOCF (Last Observed Carried Forward) imputation method.

    Attributes
    ----------
    nan : int/float
        Value used to impute data missing at the beginning of the sequence.
    """

    def __init__(self, nan=0):
        super().__init__('cpu')
        self.nan = nan

    def fit(self, train_X, val_X=None):
        warnings.warn(
            'LOCF (Last Observed Carried Forward) imputation class has no parameter to train. '
            'Please run func impute(X) directly.'
        )

    def locf_numpy(self, X):
        """ Numpy implementation of LOCF.

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

    def locf_torch(self, X):
        """ Torch implementation of LOCF.

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
        idx = torch.where(~mask, torch.arange(n_features), 0)
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

    def impute(self, X):
        """ Impute missing values

        Parameters
        ----------
        X : array-like,
            Time-series vectors containing missing values (NaN).

        Returns
        -------
        array-like,
            Imputed time series.
        """
        assert len(X.shape) == 3, f'Input X should have 3 dimensions [n_samples, n_steps, n_features], ' \
                                  f'but the actual shape of X: {X.shape}'
        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(X, np.ndarray):
            X_imputed = self.locf_numpy(X)
        elif isinstance(X, torch.Tensor):
            X_imputed = self.locf_torch(X).detach().cpu().numpy()
        else:
            raise TypeError('X must be type of list/np.ndarray/torch.Tensor, '
                            f'but got {type(X)}')
        return X_imputed
