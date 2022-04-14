"""
Implementation of the imputation method LOCF (Last Observed Carried Forward).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import warnings

import numpy as np

from pypots.imputation.base import BaseImputer


class LOCF(BaseImputer):
    """ LOCF (Last Observed Carried Forward) imputation method.

    Attributes
    ----------
    nan : int/float
        Value used to impute data missing at the beginning of the sequence.
    """

    def __init__(self, nan=0):
        super(LOCF, self).__init__('cpu')
        self.nan = nan

    def fit(self, train_X, val_X=None):
        warnings.warn(
            'LOCF (Last Observed Carried Forward) imputation class has no parameter to train. '
            'Please run func impute(X) directly.'
        )

    @staticmethod
    def locf(X):
        assert len(X.shape) == 3, f'Input X should have 3 dimensions [n_samples, seq_len, n_features], ' \
                                  f'but the actual shape of X: {X.shape}'

        trans_X = X.transpose((0, 2, 1))
        mask = np.isnan(trans_X)
        n_samples, seq_len, n_features = mask.shape
        idx = np.where(~mask, np.arange(n_features), 0)
        np.maximum.accumulate(idx, axis=2, out=idx)
        collector = []
        for x, i in zip(trans_X, idx):
            collector.append(x[np.arange(seq_len)[:, None], i])
        out = np.asarray(collector)
        out = out.transpose((0, 2, 1))
        return out

    def impute(self, X):
        """ Impute missing values

        Parameters
        ----------
        X : array,
            Time-series vectors.

        Returns
        -------
        array,
            Imputed vectors.

        """

        imputed_X = self.locf(X)

        # If there are values still missing,
        # they are missing at the beginning of the time-series sequence.
        # Impute them with self.nan
        if np.isnan(imputed_X).any():
            imputed_X = np.nan_to_num(imputed_X, nan=self.nan)

        return imputed_X
