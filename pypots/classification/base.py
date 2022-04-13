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
    def fit(self, X, y):
        """ Train the imputer.

        Parameters
        ----------
        X : array-like of shape [n_samples, sequence length (time steps), n_features],
            Time-series data for training, can contain missing values.

        y : array,
            Classification labels.

        Returns
        -------
        self : object,
            Trained imputer.
        """
        return self

    @abstractmethod
    def classify(self, X, y):
        """ Impute missing data with the trained model.

        Parameters
        ----------
        X : array-like of shape [n_samples, sequence length (time steps), n_features],
            Time-series data for imputing contains missing values.

        y : array,
            Classification labels.

        Returns
        -------
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        pass
