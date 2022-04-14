"""
The base class for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


from abc import abstractmethod

from pypots.base import BaseModel


class BaseClassifier(BaseModel):
    """ Abstract class for all classification models.
    """

    def __init__(self, device):
        super(BaseClassifier, self).__init__(device)

    @abstractmethod
    def fit(self, train_X, train_y, val_X=None, val_y=None):
        """ Train the imputer.

        Parameters
        ----------
        train_X : array-like of shape [n_samples, sequence length (time steps), n_features],
            Time-series data for training, can contain missing values.
        train_y : array,
            Classification labels for training.
        val_X : array-like of shape [n_samples, sequence length (time steps), n_features],
            Time-series data for validation, can contain missing values.
        val_y : array,
            Classification labels for validation.

        Returns
        -------
        self : object,
            Trained imputer.
        """
        return self

    @abstractmethod
    def classify(self, X):
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
