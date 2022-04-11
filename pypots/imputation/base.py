"""
The base class for imputation models.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


from abc import ABC, abstractmethod

import torch


class BaseImputer(ABC):
    """ Abstract class for all imputation models.
    """

    @abstractmethod
    def __init__(self):
        pass

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

    def _train_model(self, training_loader):
        """ Model training procedure.
        Parameters
        ----------
        training_loader : torch.utils.data.Dataset object
            Data loader for training

        """
        pass

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

    def save_model(self, saving_path):
        """ Save the model to a disk file.

        Parameters
        ----------
        saving_path : the given path to save the model
        """
        try:
            torch.save(self, saving_path)
        except Exception as e:
            print(e)
        print(f'Saved successfully to {saving_path}.')

    @staticmethod
    def load_model(file_path):
        """ Load model from a disk file.

        Parameters
        ----------
        file_path : the path to a pre-saved model file.

        Returns
        -------
        object,
            Loaded model object.

        """
        return torch.load(file_path)
