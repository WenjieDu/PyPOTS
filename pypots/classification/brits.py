"""
PyTorch BRITS model for both the time-series imputation task and the classification task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.classification.base import BaseNNClassifier
from pypots.data import DatasetForBRITS
from pypots.imputation.brits import (
    RITS as imputation_RITS,
    _BRITS as imputation_BRITS
)


class RITS(imputation_RITS):
    def __init__(self, n_steps, n_features, rnn_hidden_size, n_classes, device=None):
        super().__init__(n_steps, n_features, rnn_hidden_size, device)
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(self.rnn_hidden_size, n_classes)

    def forward(self, inputs, direction='forward'):
        ret_dict = super().forward(inputs, direction)
        logits = self.classifier(ret_dict['final_hidden_state'])
        ret_dict['prediction'] = torch.softmax(logits, dim=1)
        return ret_dict


class _BRITS(imputation_BRITS, nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size, n_classes,
                 classification_weight, reconstruction_weight, device=None):
        super().__init__(n_steps, n_features, rnn_hidden_size)
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, n_classes, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, n_classes, device)
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

    def merge_ret(self, ret_f, ret_b):
        """ Merge (average) results from two RITS models into one.

        Parameters
        ----------
        ret_f : dict,
            Results from the forward RITS.
        ret_b : dict,
            Results from the backward RITS.

        Returns
        -------
        dict,
            Merged results in a dictionary.
        """
        results = {
            'imputed_data': (ret_f['imputed_data'] + ret_b['imputed_data']) / 2,
            'prediction': (ret_f['prediction'] + ret_b['prediction']) / 2
        }
        return results

    def classify(self, inputs):
        ret_f = self.rits_f(inputs, 'forward')
        ret_b = self.reverse(self.rits_b(inputs, 'backward'))
        merged_ret = self.merge_ret(ret_f, ret_b)
        return merged_ret, ret_f, ret_b

    def forward(self, inputs):
        """ Forward processing of BRITS.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        merged_ret, ret_f, ret_b = self.classify(inputs)
        ret_f['classification_loss'] = F.nll_loss(torch.log(ret_f['prediction']), inputs['label'])
        ret_b['classification_loss'] = F.nll_loss(torch.log(ret_b['prediction']), inputs['label'])
        consistency_loss = self.get_consistency_loss(ret_f['imputed_data'], ret_b['imputed_data'])
        classification_loss = (ret_f['classification_loss'] + ret_b['classification_loss']) / 2
        merged_ret['consistency_loss'] = consistency_loss
        merged_ret['classification_loss'] = classification_loss
        merged_ret['loss'] = \
            consistency_loss + \
            (ret_f['reconstruction_loss'] + ret_b['reconstruction_loss']) * self.reconstruction_weight + \
            (ret_f['classification_loss'] + ret_b['classification_loss']) * self.classification_weight
        return merged_ret


class BRITS(BaseNNClassifier):
    """ BRITS implementation of BaseClassifier.

    Attributes
    ----------
    model : object,
        The underlying BRITS model.
    optimizer : object,
        The optimizer for model training.
    data_loader : object,
        The data loader for dataset loading.

    Parameters
    ----------
    rnn_hidden_size : int,
        The size of the RNN hidden state.
    learning_rate : float (0,1),
        The learning rate parameter for the optimizer.
    weight_decay : float in (0,1),
        The weight decay parameter for the optimizer.
    epochs : int,
        The number of training epochs.
    patience : int,
        The number of epochs with loss non-decreasing before early stopping the training.
    batch_size : int,
        The batch size of the training input.
    device :
        Run the model on which device.
    """

    def __init__(self,
                 n_steps,
                 n_features,
                 rnn_hidden_size,
                 n_classes,
                 classification_weight=1,
                 reconstruction_weight=1,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super().__init__(n_classes, learning_rate, epochs, patience, batch_size, weight_decay, device)

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        self.model = _BRITS(self.n_steps, self.n_features, self.rnn_hidden_size, self.n_classes,
                            self.classification_weight, self.reconstruction_weight, self.device)
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_X, train_y, val_X=None, val_y=None):
        """ Fit the model on the given training data.

        Parameters
        ----------
        train_X : array, shape [n_samples, sequence length (time steps), n_features],
            Time-series vectors.
        train_y : array,
            Classification labels.

        Returns
        -------
        self : object,
            Trained model.
        """
        train_X, train_y = self.check_input(self.n_steps, self.n_features, train_X, train_y)
        val_X, val_y = self.check_input(self.n_steps, self.n_features, val_X, val_y)

        training_set = DatasetForBRITS(train_X, train_y)  # time_gaps is necessary for BRITS
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)

        if val_X is None:
            self._train_model(training_loader)
        else:
            val_set = DatasetForBRITS(val_X, val_y)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(training_loader, val_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.
        return self

    def assemble_input_data(self, data):
        """ Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        # fetch data
        indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, label = data

        # assemble input data
        inputs = {
            'indices': indices,
            'label': label,
            'forward': {
                'X': X,
                'missing_mask': missing_mask,
                'deltas': deltas
            },
            'backward': {
                'X': back_X,
                'missing_mask': back_missing_mask,
                'deltas': back_deltas
            }
        }
        return inputs

    def classify(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForBRITS(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        prediction_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                # cannot use input_data_processing, cause here has no label
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = data
                # assemble input data
                inputs = {
                    'indices': indices,
                    'forward': {
                        'X': X,
                        'missing_mask': missing_mask,
                        'deltas': deltas
                    },
                    'backward': {
                        'X': back_X,
                        'missing_mask': back_missing_mask,
                        'deltas': back_deltas
                    }
                }

                results, _, _ = self.model.classify(inputs)
                prediction_collector.append(results['prediction'])

        predictions = torch.cat(prediction_collector)
        return predictions.cpu().detach().numpy()
