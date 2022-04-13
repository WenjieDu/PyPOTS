"""
The base class for imputation models.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


from abc import abstractmethod

from pypots.base import BaseModel


class BaseImputer(BaseModel):
    """ Abstract class for all imputation models.
    """

    def __init__(self, device):
        super(BaseImputer, self).__init__(device)

    @abstractmethod
    def fit(self, X):
        """ Train the imputer.

        Parameters
        ----------
        X : array-like of shape [n_samples, sequence length (time steps), n_features],
            Time-series data for training, can contain missing values.

        Returns
        -------
        self : object,
            Trained imputer.
        """
        return self

    @abstractmethod
    def impute(self, X):
        """ Impute missing data with the trained model.

        Parameters
        ----------
        X : array-like of shape [n_samples, sequence length (time steps), n_features],
            Time-series data for imputing contains missing values.

        Returns
        -------
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        pass
