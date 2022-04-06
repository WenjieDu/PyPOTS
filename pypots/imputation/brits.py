"""
PyTorch BRITS model for the time-series imputation task.
Some part of the code is from https://github.com/caow13/BRITS.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from pypots.datasets.base import BaseDataset
from pypots.imputation.base import BaseImputer
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
        super(FeatureRegression, self).__init__()
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
        super(TemporalDecay, self).__init__()
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
    seq_len : int,
        sequence length (number of time steps)
    feature_num : int,
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
    seq_len : int,
        sequence length (number of time steps)
    feature_num : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    device : str,
        specify running the model on which device, CPU/GPU
    """

    def __init__(self, seq_len, feature_num, rnn_hidden_size, device=None):
        super(RITS, self).__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.rnn_cell = nn.LSTMCell(self.feature_num * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.feature_num, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.feature_num, output_size=self.feature_num, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.feature_num)
        self.feat_reg = FeatureRegression(self.feature_num)
        self.combining_weight = nn.Linear(self.feature_num * 2, self.feature_num)

    def impute(self, data, direction):
        """ The imputation function.
        Parameters
        ----------
        data : dict,
            Input data, a dictionary includes feature values, missing masks, and time-gap values.
        direction : str, 'forward'/'backward'
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data : tensor,
            [batch size, sequence length, feature number]
        hidden_states: tensor,
            [batch size, RNN hidden size]
        reconstruction_MAE : float tensor,
            mean absolute error of reconstruction
        reconstruction_loss : float tensor,
            reconstruction loss
        """
        values = data[direction]['X']  # feature values
        masks = data[direction]['missing_mask']  # missing masks
        deltas = data[direction]['deltas']  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)
        cell_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)

        estimations = []
        reconstruction_loss = 0.0
        reconstruction_MAE = 0.0

        # imputation period
        for t in range(self.seq_len):
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

            alpha = F.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_MAE += cal_mae(c_h, x, m)
            reconstruction_loss += reconstruction_MAE

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(inputs, (hidden_states, cell_states))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, hidden_states, reconstruction_MAE, reconstruction_loss

    def forward(self, data, direction='forward'):
        """ Forward processing of the NN module.
        Parameters
        ----------
        data : dict, the input data
        direction : a keyword to extract data from parameter `data`, 'forward'/'backward'

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        imputed_data, hidden_state, reconstruction_MAE, reconstruction_loss = self.impute(data, direction)
        reconstruction_MAE /= self.seq_len
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= (self.seq_len * 3)

        ret_dict = {
            'consistency_loss': torch.tensor(0.0, device=self.device),  # single direction, has no consistency loss
            'reconstruction_loss': reconstruction_loss,
            'reconstruction_MAE': reconstruction_MAE,
            'imputed_data': imputed_data,
            'final_hidden_state': hidden_state
        }
        return ret_dict


class _BRITS(nn.Module):
    """ model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    seq_len : int,
        sequence length (number of time steps)
    feature_num : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    rits_f: RITS object
        the forward RITS model
    rits_b: RITS object
        the backward RITS model
    """

    def __init__(self, seq_len, feature_num, rnn_hidden_size, device=None):
        super(_BRITS, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(seq_len, feature_num, rnn_hidden_size, device)
        self.rits_b = RITS(seq_len, feature_num, rnn_hidden_size, device)

    def impute(self, data):
        """ Impute the missing data. Only impute, this is for test stage.

        Parameters
        ----------
        data : dict,
            The dictionary includes all input data.

        Returns
        -------
        array-like, the same shape with the input feature vectors.
            The feature vectors with missing part imputed.

        """
        imputed_data_f, _, _, _ = self.rits_f.impute(data, 'forward')
        imputed_data_b, _, _, _ = self.rits_b.impute(data, 'backward')
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
        reconstruction_MAE = (ret_f['reconstruction_MAE'] + ret_b['reconstruction_MAE']) / 2
        ret_f['imputed_data'] = (ret_f['imputed_data'] + ret_b['imputed_data']) / 2
        ret_f['consistency_loss'] = consistency_loss
        ret_f['reconstruction_MAE'] = reconstruction_MAE
        ret_f['loss'] = consistency_loss + \
                        ret_f['reconstruction_loss'] + \
                        ret_b['reconstruction_loss']

        return ret_f

    def forward(self, data):
        """ Forward processing of BRITS.

        Parameters
        ----------
        data : dict,
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))
        ret = self.merge_ret(ret_f, ret_b)
        return ret


def parse_delta(missing_mask):
    """ Generate time-gap (delta) matrix from missing masks.

    Parameters
    ----------
    missing_mask : array, shape of [seq_len, n_features]
        Binary masks indicate missing values.

    Returns
    -------
    array,
        Delta matrix indicates time gaps of missing values.
        Its math definition please refer to :cite:`che2018MissingData`.
    """
    assert len(missing_mask.shape) == 2, f'missing_mask should has two dimensions, ' \
                                         f'shape like [seq_len, n_features], ' \
                                         f'while the input is {missing_mask.shape}'
    seq_len, n_features = missing_mask.shape
    delta = []
    for step in range(seq_len):
        if step == 0:
            delta.append(np.zeros(n_features))
        else:
            delta.append(np.ones(n_features) + (1 - missing_mask[step]) * delta[-1])
    return np.asarray(delta)


class Dataset4BRITS(BaseDataset):
    """ Dataset class for BRITS.

    Parameters
    ----------
    X : array-like, shape of [n_samples, seq_len, n_features]
        Time-series feature vector.
    y : array-like, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.
    """

    def __init__(self, X, y=None):
        super(Dataset4BRITS, self).__init__(X, y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """ Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        dict,
            A dict contains
            index : int tensor,
                The index of the sample.
            X : tensor,
                The feature vector for model input.
            missing_mask : tensor,
                The mask indicates all missing values in X.
            delta : tensor,
                The delta matrix contains time gaps of missing values.
            label (optional) : tensor,
                The target label of the time-series sample.
        """
        X = self.X[idx]
        missing_mask = (~np.isnan(X)).astype(np.float32)
        X = np.nan_to_num(X)

        forward = {'X': X, 'missing_mask': missing_mask, 'deltas': parse_delta(missing_mask)}
        backward = {'X': np.flip(forward['X'], axis=0).copy(),
                    'missing_mask': np.flip(forward['missing_mask'], axis=0).copy()}
        backward['deltas'] = parse_delta(backward['missing_mask'])
        sample = [
            torch.tensor(idx),
            # for forward
            torch.from_numpy(forward['X'].astype('float32')),
            torch.from_numpy(forward['missing_mask'].astype('float32')),
            torch.from_numpy(forward['deltas'].astype('float32')),
            # for backward
            torch.from_numpy(backward['X'].astype('float32')),
            torch.from_numpy(backward['missing_mask'].astype('float32')),
            torch.from_numpy(backward['deltas'].astype('float32')),
        ]

        return sample


class BRITS(BaseImputer):
    """ BRITS implementation of BaseImputer

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
    seq_len : int,
        Sequence length/ time steps of the input.
    feature_num : int,
        Features number of the input data.
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
                 seq_len,
                 feature_num,
                 rnn_hidden_size,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super(BRITS, self).__init__()

        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = learning_rate
        self.weight_decay = weight_decay
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = None
        self.optimizer = None
        self.data_loader = None

    def fit(self, X):
        """ Fit the model on the given training data.

        Parameters
        ----------
        array-like of shape [n_samples, sequence length (time steps), n_features],

        Returns
        -------
        self : object,
            Trained model.
        """

        self.model = _BRITS(self.seq_len, self.feature_num, self.rnn_hidden_size, self.device)
        self.model = self.model.to(self.device)
        training_set = Dataset4BRITS(X)  # time_gaps is necessary for BRITS
        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self._train_model(train_loader)
        self.model.load_state_dict(self.best_model_dict)
        return self

    def _train_model(self, training_loader):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        self.best_loss = float('inf')
        self.best_model_dict = None

        self.model.train()
        for epoch in range(self.epochs):
            loss_collector = []
            for idx, data in enumerate(training_loader):
                # fetch data
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                    map(lambda x: x.to(self.device), data)
                # assemble input data
                inputs = {'indices': indices,
                          'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}
                          }
                self.optimizer.zero_grad()
                results = self.model.forward(inputs)
                results['loss'].backward()
                self.optimizer.step()
                loss_collector.append(results['loss'].item())
            print('epoch {epoch}: training loss {train_loss} '.format(
                epoch=epoch, train_loss=np.mean(loss_collector)))

        self.model.eval()

    def impute(self, X):
        return self.model.impute(X)
