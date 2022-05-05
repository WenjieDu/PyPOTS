"""
PyTorch BRITS model for the time-series imputation task.
Some part of the code is from https://github.com/caow13/BRITS.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pycorruptor import mcar, fill_nan_with_mask
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from pypots.data.dataset_for_brits import DatasetForBRITS
from pypots.imputation.base import BaseNNImputer
from pypots.utils.metrics import cal_mae


class FeatureRegression(nn.Module):
    """ The module used to capture the correlation between features for imputation.

    Attributes
    ----------
    W : tensor
        The weights (parameters) of the module.
    b : tensor
        The bias of the module.
    m (buffer) : tensor
        The mask matrix, a squire matrix with diagonal entries all zeroes while left parts all ones.
        It is applied to the weight matrix to mask out the estimation contributions from features themselves.
        It is used to help enhance the imputation performance of the network.

    Parameters
    ----------
    input_size : the feature dimension of the input
    """

    def __init__(self, input_size):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ Forward processing of the NN module.

        Parameters
        ----------
        x : tensor,
            the input for processing

        Returns
        -------
        output: tensor,
            the processed result containing imputation from feature regression

        """
        output = F.linear(x, self.W * Variable(self.m), self.b)
        return output


class TemporalDecay(nn.Module):
    """ The module used to generate the temporal decay factor gamma in the original paper.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input
    output_size : int,
        the feature dimension of the output
    diag : bool,
        whether to product the weight with an identity matrix before forward processing
    """

    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, delta):
        """ Forward processing of the NN module.

        Parameters
        ----------
        delta : tensor, shape [batch size, sequence length, feature number]
            The time gaps.

        Returns
        -------
        gamma : array-like, same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    """ model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps : int,
        sequence length (number of time steps)
    n_features : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    device : str, default=None,
        specify running the model on which device, CPU/GPU
    rnn_cell : torch.nn.module object
        the LSTM cell to model temporal data
    temp_decay_h : torch.nn.module object
        the temporal decay module to decay RNN hidden state
    temp_decay_x : torch.nn.module object
        the temporal decay module to decay data in the raw feature space
    hist_reg : torch.nn.module object
        the temporal-regression module to project RNN hidden state into the raw feature space
    feat_reg : torch.nn.module object
        the feature-regression module
    combining_weight : torch.nn.module object
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps : int,
        sequence length (number of time steps)
    n_features : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    device : str,
        specify running the model on which device, CPU/GPU
    """

    def __init__(self, n_steps, n_features, rnn_hidden_size, device=None):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def impute(self, inputs, direction):
        """ The imputation function.
        Parameters
        ----------
        inputs : dict,
            Input data, a dictionary includes feature values, missing masks, and time-gap values.
        direction : str, 'forward'/'backward'
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data : tensor,
            [batch size, sequence length, feature number]
        hidden_states: tensor,
            [batch size, RNN hidden size]
        reconstruction_loss : float tensor,
            reconstruction loss
        """
        values = inputs[direction]['X']  # feature values
        masks = inputs[direction]['missing_mask']  # missing masks
        deltas = inputs[direction]['deltas']  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)
        cell_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)

        estimations = []
        reconstruction_loss = 0.0

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += cal_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += cal_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += cal_mae(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(inputs, (hidden_states, cell_states))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, hidden_states, reconstruction_loss

    def forward(self, inputs, direction='forward'):
        """ Forward processing of the NN module.
        Parameters
        ----------
        inputs : dict,
            The input data.

        direction : string, 'forward'/'backward'
            A keyword to extract data from parameter `data`.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        imputed_data, hidden_state, reconstruction_loss = self.impute(inputs, direction)
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= (self.n_steps * 3)

        ret_dict = {
            'consistency_loss': torch.tensor(0.0, device=self.device),  # single direction, has no consistency loss
            'reconstruction_loss': reconstruction_loss,
            'imputed_data': imputed_data,
            'final_hidden_state': hidden_state
        }
        return ret_dict


class _BRITS(nn.Module):
    """ model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps : int,
        sequence length (number of time steps)
    n_features : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    rits_f: RITS object
        the forward RITS model
    rits_b: RITS object
        the backward RITS model
    """

    def __init__(self, n_steps, n_features, rnn_hidden_size, device=None):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, device)

    def impute(self, inputs):
        """ Impute the missing data. Only impute, this is for test stage.

        Parameters
        ----------
        inputs : dict,
            A dictionary includes all input data.

        Returns
        -------
        array-like, the same shape with the input feature vectors.
            The feature vectors with missing part imputed.

        """
        imputed_data_f, _, _ = self.rits_f.impute(inputs, 'forward')
        imputed_data_b, _, _ = self.rits_b.impute(inputs, 'backward')
        imputed_data_b = {'imputed_data_b': imputed_data_b}
        imputed_data_b = self.reverse(imputed_data_b)['imputed_data_b']
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data

    @staticmethod
    def get_consistency_loss(pred_f, pred_b):
        """ Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f : array-like,
            The imputation from the forward RITS.
        pred_b : array-like,
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.
        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def reverse(ret):
        """ Reverse the array values on the time dimension in the given dictionary.

        Parameters
        ----------
        ret : dict

        Returns
        -------
        dict,
            A dictionary contains values reversed on the time dimension from the given dict.
        """

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

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
        consistency_loss = self.get_consistency_loss(ret_f['imputed_data'], ret_b['imputed_data'])
        ret_f['imputed_data'] = (ret_f['imputed_data'] + ret_b['imputed_data']) / 2
        ret_f['consistency_loss'] = consistency_loss
        ret_f['loss'] = consistency_loss + \
                        ret_f['reconstruction_loss'] + \
                        ret_b['reconstruction_loss']

        return ret_f

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
        ret_f = self.rits_f(inputs, 'forward')
        ret_b = self.reverse(self.rits_b(inputs, 'backward'))
        ret = self.merge_ret(ret_f, ret_b)
        return ret


class BRITS(BaseNNImputer):
    """ BRITS implementation

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
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super().__init__(learning_rate, epochs, patience, batch_size, weight_decay, device)

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.model = _BRITS(self.n_steps, self.n_features, self.rnn_hidden_size, self.device)
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_X, val_X=None):
        """ Fit the model on the given training data.

        Parameters
        ----------
        train_X : array-like, shape of [n_samples, n_steps, n_features],
            Data for training.

        val_X : array-like, optional, shape of [n_samples, n_steps, n_features],
            Data for validating.

        Returns
        -------
        self : object,
            Trained model.
        """
        train_X = self.check_input(self.n_steps, self.n_features, train_X)
        if val_X is not None:
            val_X = self.check_input(self.n_steps, self.n_features, val_X)

        training_set = DatasetForBRITS(train_X)  # time_gaps is necessary for BRITS
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)

        if val_X is None:
            self._train_model(training_loader)
        else:
            val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(val_X, 0.2)
            val_X = fill_nan_with_mask(val_X, val_X_missing_mask)
            val_set = DatasetForBRITS(val_X)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(training_loader, val_loader, val_X_intact, val_X_indicating_mask)

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

        return inputs

    def impute(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForBRITS(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self.assemble_input_data(data)
                imputed_data = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
