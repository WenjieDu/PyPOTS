"""
The base class for classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


from abc import abstractmethod

import numpy as np
import torch

from pypots.base import BaseModel, BaseNNModel


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


class BaseNNClassifier(BaseNNModel, BaseClassifier):
    def __init__(self, n_classes, learning_rate, epochs, patience, batch_size, weight_decay, device):
        super().__init__(learning_rate, epochs, patience, batch_size, weight_decay, device)
        self.n_classes = n_classes

    @abstractmethod
    def input_data_processing(self, data):
        pass

    def _train_model(self, training_loader, val_loader=None):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss_collector = []
            for idx, data in enumerate(training_loader):
                inputs = self.input_data_processing(data)
                self.optimizer.zero_grad()
                results = self.model.forward(inputs)
                results['loss'].backward()
                self.optimizer.step()
                epoch_train_loss_collector.append(results['loss'].item())

            mean_train_loss = np.mean(epoch_train_loss_collector)  # mean training loss of the current epoch
            self.logger['training_loss'].append(mean_train_loss)

            if val_loader is not None:
                self.model.eval()
                epoch_val_loss_collector = []
                with torch.no_grad():
                    for idx, data in enumerate(val_loader):
                        inputs = self.input_data_processing(data)
                        results = self.model.forward(inputs)
                        epoch_val_loss_collector.append(results['loss'].item())

                mean_val_loss = np.mean(epoch_train_loss_collector)
                self.logger['validating_loss'].append(mean_val_loss)
                print(f'epoch {epoch}: training loss {mean_train_loss:.4f}, validating loss {mean_val_loss:.4f}')
                mean_loss = mean_val_loss
            else:
                print(f'epoch {epoch}: training loss {mean_train_loss:.4f}')
                mean_loss = mean_train_loss

            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.best_model_dict = self.model.state_dict()
            else:
                self.patience -= 1
                if self.patience == 0:
                    print('Exceeded the training patience. Terminating the training procedure...')
                    break
        print('Finished training.')
