"""
Torch implementation of CRLI (Clustering Representation Learning on Incomplete time-series data).

Please refer to :cite:``ma2021CRLI``.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from pypots.clustering.base import BaseNNClusterer
from pypots.data.dataset_for_grud import DatasetForGRUD
from pypots.utils.metrics import cal_mse

RNN_CELL = {
    'LSTM': nn.LSTMCell,
    'GRU': nn.GRUCell,
}


def reverse_tensor(tensor_):
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
    return tensor_.index_select(1, indices)


class MultiRNNCell(nn.Module):
    def __init__(self, cell_type, n_layer, d_input, d_hidden, device):
        super(MultiRNNCell, self).__init__()
        self.cell_type = cell_type
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.device = device

        self.model = nn.ModuleList()
        if cell_type in ['LSTM', 'GRU']:
            for i in range(n_layer):
                if i == 0:
                    self.model.append(RNN_CELL[cell_type](d_input, d_hidden))
                else:
                    self.model.append(RNN_CELL[cell_type](d_hidden, d_hidden))

        self.output_layer = nn.Linear(d_hidden, d_input)

    def forward(self, inputs):
        X = inputs['X']
        bz, n_steps, _ = X.shape
        hidden_state = torch.zeros((bz, self.d_hidden), device=self.device)
        hidden_state_collector = torch.empty((bz, n_steps, self.d_hidden), device=self.device)
        # output_collector = torch.empty((bz, n_steps, self.d_input), device=self.device)
        if self.cell_type == 'LSTM':
            # TODO: cell states should have different shapes
            cell_states = torch.zeros((self.d_input, self.d_hidden), device=self.device)
            for step in range(n_steps):
                x = X[:, step, :]
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state, cell_states = self.model[i](x, (hidden_state, cell_states))
                    else:
                        hidden_state, cell_states = self.model[i](hidden_state, (hidden_state, cell_states))

                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == 'GRU':
            for step in range(n_steps):
                x = X[:, step, :]
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state = self.model[i](x, hidden_state)
                    else:
                        hidden_state = self.model[i](hidden_state, hidden_state)

                hidden_state_collector[:, step, :] = hidden_state

        output_collector = self.output_layer(hidden_state_collector)
        return output_collector, hidden_state


class Generator(nn.Module):
    def __init__(self, n_layers, n_features, d_hidden, cell_type, device):
        super().__init__()
        self.f_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden, device)
        self.b_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden, device)

    def forward(self, inputs):
        f_outputs, f_final_hidden_state = self.f_rnn(inputs)
        b_outputs, b_final_hidden_state = self.b_rnn(inputs)
        b_outputs = reverse_tensor(b_outputs)  # reverse the output of the backward rnn
        imputation = (f_outputs + b_outputs) / 2
        imputed_X = inputs['X'] * inputs['missing_mask'] + imputation * (1 - inputs['missing_mask'])
        fb_final_hidden_states = torch.concat([f_final_hidden_state, b_final_hidden_state], dim=-1)
        return imputation, imputed_X, fb_final_hidden_states


class Discriminator(nn.Module):
    def __init__(self, cell_type, d_input, device='cpu'):
        super().__init__()
        self.cell_type = cell_type
        self.device = device
        # this setting is the same with the official implementation
        self.rnn_cell_module_list = nn.ModuleList([
            RNN_CELL[cell_type](d_input, 32),
            RNN_CELL[cell_type](32, 16),
            RNN_CELL[cell_type](16, 8),
            RNN_CELL[cell_type](8, 16),
            RNN_CELL[cell_type](16, 32),
        ])
        self.output_layer = nn.Linear(32, d_input)

    def forward(self, inputs):
        imputed_X = inputs['imputed_X']
        bz, n_steps, _ = imputed_X.shape
        hidden_states = [
            torch.zeros((bz, 32), device=self.device),
            torch.zeros((bz, 16), device=self.device),
            torch.zeros((bz, 8), device=self.device),
            torch.zeros((bz, 16), device=self.device),
            torch.zeros((bz, 32), device=self.device)
        ]
        hidden_state_collector = torch.empty((bz, n_steps, 32), device=self.device)
        if self.cell_type == 'LSTM':
            cell_states = torch.zeros((self.d_input, self.d_hidden), device=self.device)
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state, cell_states = rnn_cell(x, (hidden_states[i], cell_states))
                    else:
                        hidden_state, cell_states = rnn_cell(hidden_states[i - 1], (hidden_states[i], cell_states))
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == 'GRU':
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state = rnn_cell(x, hidden_states[i])
                    else:
                        hidden_state = rnn_cell(hidden_states[i - 1], hidden_states[i])
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state

        output_collector = self.output_layer(hidden_state_collector)
        return output_collector


class Decoder(nn.Module):
    def __init__(self, n_steps, d_input, d_output, fcn_output_dims: list = None, device='cpu'):
        super().__init__()
        self.n_steps = n_steps
        self.d_output = d_output
        self.device = device

        if fcn_output_dims is None:
            fcn_output_dims = [d_input]
        self.fcn_output_dims = fcn_output_dims

        self.fcn = nn.ModuleList()
        for output_dim in fcn_output_dims:
            self.fcn.append(nn.Linear(d_input, output_dim))
            d_input = output_dim

        self.rnn_cell = nn.GRUCell(fcn_output_dims[-1], fcn_output_dims[-1])
        self.output_layer = nn.Linear(fcn_output_dims[-1], d_output)

    def forward(self, inputs):
        generator_fb_hidden_states = inputs['generator_fb_hidden_states']
        bz, _ = generator_fb_hidden_states.shape
        fcn_latent = generator_fb_hidden_states
        for layer in self.fcn:
            fcn_latent = layer(fcn_latent)
        hidden_state = fcn_latent
        hidden_state_collector = torch.empty((bz, self.n_steps, self.fcn_output_dims[-1]), device=self.device)
        for i in range(self.n_steps):
            hidden_state = self.rnn_cell(hidden_state, hidden_state)
            hidden_state_collector[:, i, :] = hidden_state
        reconstruction = self.output_layer(hidden_state_collector)
        return reconstruction, fcn_latent


class _CRLI(nn.Module):
    def __init__(self, n_steps, n_features, n_clusters, n_generator_layers, rnn_hidden_size, decoder_fcn_output_dims,
                 lambda_kmeans, rnn_cell_type='GRU', device='cpu'):
        super().__init__()
        self.generator = Generator(n_generator_layers, n_features, rnn_hidden_size, rnn_cell_type, device)
        self.discriminator = Discriminator(rnn_cell_type, n_features, device)
        self.decoder = Decoder(
            n_steps, rnn_hidden_size * 2, n_features, decoder_fcn_output_dims, device
        )  # fully connected network is included in Decoder
        self.kmeans = KMeans(n_clusters=n_clusters)  # TODO: implement KMean with torch for gpu acceleration

        self.n_clusters = n_clusters
        self.lambda_kmeans = lambda_kmeans
        self.device = device

    def cluster(self, inputs, training_object='generator'):
        # concat final states from generator and input it as the initial state of decoder
        imputation, imputed_X, generator_fb_hidden_states = self.generator(inputs)
        inputs['imputation'] = imputation
        inputs['imputed_X'] = imputed_X
        inputs['generator_fb_hidden_states'] = generator_fb_hidden_states
        if training_object == 'discriminator':
            discrimination = self.discriminator(inputs)
            inputs['discrimination'] = discrimination
            return inputs  # if only train discriminator, then no need to run decoder

        reconstruction, fcn_latent = self.decoder(inputs)
        inputs['reconstruction'] = reconstruction
        inputs['fcn_latent'] = fcn_latent
        return inputs

    def forward(self, inputs, training_object='generator'):
        assert training_object in ['generator', 'discriminator'], \
            'training_object should be "generator" or "discriminator"'

        X = inputs['X']
        missing_mask = inputs['missing_mask']
        batch_size, n_steps, n_features = X.shape
        losses = {}
        inputs = self.cluster(inputs, training_object)
        if training_object == 'discriminator':
            l_D = F.binary_cross_entropy_with_logits(inputs['discrimination'], missing_mask)
            losses['l_disc'] = l_D
        else:
            inputs['discrimination'] = inputs['discrimination'].detach()
            l_G = F.binary_cross_entropy_with_logits(inputs['discrimination'], 1 - missing_mask,
                                                     weight=1 - missing_mask)
            l_pre = cal_mse(inputs['imputation'], X, missing_mask)
            l_rec = cal_mse(inputs['reconstruction'], X, missing_mask)
            HTH = torch.matmul(inputs['fcn_latent'], inputs['fcn_latent'].permute(1, 0))
            term_F = torch.nn.init.orthogonal_(
                torch.randn(batch_size, self.n_clusters, device=self.device),
                gain=1
            )
            FTHTHF = torch.matmul(torch.matmul(term_F.permute(1, 0), HTH), term_F)
            l_kmeans = torch.trace(HTH) - torch.trace(FTHTHF)  # k-means loss
            loss_gene = l_G + l_pre + l_rec + l_kmeans * self.lambda_kmeans
            losses['l_gene'] = loss_gene
        return losses


class CRLI(BaseNNClusterer):
    def __init__(self,
                 n_steps,
                 n_features,
                 n_clusters,
                 n_generator_layers,
                 rnn_hidden_size,
                 decoder_fcn_output_dims=None,
                 lambda_kmeans=1,
                 rnn_cell_type='GRU',
                 G_steps=1,
                 D_steps=1,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super().__init__(n_clusters, learning_rate, epochs, patience, batch_size, weight_decay, device)
        assert G_steps > 0 and D_steps > 0, 'G_steps and D_steps should both >0'

        self.n_steps = n_steps
        self.n_features = n_features
        self.G_steps = G_steps
        self.D_steps = D_steps

        self.model = _CRLI(n_steps, n_features, n_clusters, n_generator_layers, rnn_hidden_size,
                           decoder_fcn_output_dims, lambda_kmeans, rnn_cell_type, device)
        self.model = self.model.to(self.device)
        self._print_model_size()
        self.logger = {
            'training_loss_generator': [],
            'training_loss_discriminator': []
        }

    def fit(self, train_X):
        train_X = self.check_input(self.n_steps, self.n_features, train_X)
        training_set = DatasetForGRUD(train_X)
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        self._train_model(training_loader)
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
        indices, X, _, missing_mask, _, _ = data

        inputs = {
            'X': X,
            'missing_mask': missing_mask,
        }
        return inputs

    def _train_model(self, training_loader, val_loader=None):
        self.G_optimizer = torch.optim.Adam(
            [
                {'params': self.model.generator.parameters()},
                {'params': self.model.decoder.parameters()}
            ],
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.D_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr,
                                            weight_decay=self.weight_decay)

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float('inf')
        self.best_model_dict = None

        try:
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss_G_collector = []
                epoch_train_loss_D_collector = []
                for idx, data in enumerate(training_loader):
                    inputs = self.assemble_input_data(data)

                    for _ in range(self.D_steps):
                        self.D_optimizer.zero_grad()
                        results = self.model.forward(inputs, training_object='discriminator')
                        results['l_disc'].backward(retain_graph=True)
                        self.D_optimizer.step()
                        epoch_train_loss_D_collector.append(results['l_disc'].item())

                    for _ in range(self.G_steps):
                        self.G_optimizer.zero_grad()
                        results = self.model.forward(inputs, training_object='generator')
                        results['l_gene'].backward()
                        self.G_optimizer.step()
                        epoch_train_loss_G_collector.append(results['l_gene'].item())

                mean_train_G_loss = np.mean(epoch_train_loss_G_collector)  # mean training loss of the current epoch
                mean_train_D_loss = np.mean(epoch_train_loss_D_collector)  # mean training loss of the current epoch
                self.logger['training_loss_generator'].append(mean_train_G_loss)
                self.logger['training_loss_discriminator'].append(mean_train_D_loss)
                print(f'epoch {epoch}: '
                      f'training loss_generator {mean_train_G_loss:.4f}, '
                      f'train loss_discriminator {mean_train_D_loss:.4f}')
                mean_loss = mean_train_G_loss

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

    def cluster(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        latent_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self.assemble_input_data(data)
                inputs = self.model.cluster(inputs)
                latent_collector.append(inputs['fcn_latent'])

        latent_collector = torch.cat(latent_collector).cpu().detach().numpy()
        clustering = self.model.kmeans.fit_predict(latent_collector)

        return clustering
