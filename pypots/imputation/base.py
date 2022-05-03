"""
The base class for imputation models.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


from abc import abstractmethod

import numpy as np
import torch

from pypots.base import BaseModel, BaseNNModel
from pypots.utils.metrics import cal_mae


class BaseImputer(BaseModel):
    """ Abstract class for all imputation models.
    """

    def __init__(self, device):
        super().__init__(device)

    @abstractmethod
    def fit(self, train_X, val_X=None):
        """ Train the imputer.

        Parameters
        ----------
        train_X : array-like, shape: [n_samples, sequence length (time steps), n_features],
            Time-series data for training, can contain missing values.
        val_X : array-like, optional, shape [n_samples, sequence length (time steps), n_features],
            Time-series data for validating, can contain missing values.

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


class BaseNNImputer(BaseNNModel, BaseImputer):
    def __init__(self, learning_rate, epochs, patience, batch_size, weight_decay, device):
        super().__init__(learning_rate, epochs, patience, batch_size, weight_decay, device)

    @abstractmethod
    def assemble_input_data(self, data):
        pass

    def _train_model(self, training_loader, val_loader=None, val_X_intact=None, val_indicating_mask=None):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float('inf')
        self.best_model_dict = None

        try:
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    inputs = self.assemble_input_data(data)
                    self.optimizer.zero_grad()
                    results = self.model.forward(inputs)
                    results['loss'].backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results['loss'].item())

                mean_train_loss = np.mean(epoch_train_loss_collector)  # mean training loss of the current epoch
                self.logger['training_loss'].append(mean_train_loss)

                if val_loader is not None:
                    self.model.eval()
                    imputation_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self.assemble_input_data(data)
                            results = self.model.forward(inputs)
                            imputation_collector.append(results['imputed_data'])

                    imputation_collector = torch.cat(imputation_collector)
                    imputation_collector = imputation_collector

                    mean_val_loss = cal_mae(imputation_collector, val_X_intact, val_indicating_mask)
                    self.logger['validating_loss'].append(mean_val_loss)
                    print(f'epoch {epoch}: training loss {mean_train_loss:.4f}, validating loss {mean_val_loss:.4f}')
                    mean_loss = mean_val_loss
                else:
                    print(f'epoch {epoch}: training loss {mean_train_loss:.4f}')
                    mean_loss = mean_train_loss

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1
                    if self.patience == 0:
                        print('Exceeded the training patience. Terminating the training procedure...')
                        break
        except Exception as e:
            print(f'Exception: {e}')
            if self.best_model_dict is None:
                raise RuntimeError('Training got interrupted. Model was not get trained. Please try fit() again.')
            else:
                RuntimeWarning('Training got interrupted. '
                               'Model will load the best parameters so far for testing. '
                               "If you don't want it, please try fit() again.")

        if np.equal(self.best_loss, float('inf')):
            raise ValueError('Something is wrong. best_loss is Nan after training.')

        print('Finished training.')
