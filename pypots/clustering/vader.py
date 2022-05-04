"""
Torch implementation of model VaDER.

Refer to paper :cite:`dejong2019VaDER`.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from pypots.clustering.base import BaseNNClusterer
from pypots.data.dataset_for_grud import DatasetForGRUD
from pypots.utils.metrics import cal_mse


class ImplicitImputation(nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.projection_layer = nn.Linear(d_input, d_input, bias=False)

    def forward(self, X, missing_mask):
        imputation = self.projection_layer(X)
        imputed_X = X * missing_mask + imputation * (1 - X)
        return imputed_X


class PeepholeLSTMCell(nn.LSTMCell):
    """
    Notes
    -----
    This implementation is adapted from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.weight_ch = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ch = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(hidden_size))
        self.reset_parameters()

    def forward(self, input, hx=None):
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)

        h, c = hx

        wx = F.linear(input, self.weight_ih, self.bias_ih)
        wh = F.linear(h, self.weight_hh, self.bias_hh)
        wc = F.linear(c, self.weight_ch, self.bias_ch)

        wxhc = wx + \
               wh + \
               torch.cat(
                   (
                       wc[:, :2 * self.hidden_size],
                       Variable(self.wc_blank).expand_as(h),
                       wc[:, 2 * self.hidden_size:]
                   ),
                   dim=1
               )

        i = torch.sigmoid(wxhc[:, :self.hidden_size])
        f = torch.sigmoid(wxhc[:, self.hidden_size:2 * self.hidden_size])
        g = torch.tanh(wxhc[:, 2 * self.hidden_size:3 * self.hidden_size])
        o = torch.sigmoid(wxhc[:, 3 * self.hidden_size:])

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class GMMLayer(nn.Module):
    def __init__(self, d_hidden, n_clusters):
        super().__init__()
        self.mu_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.var_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.phi_c_unscaled = torch.Tensor(n_clusters)

    def set_values(self, mu, var, phi):
        assert mu.shape == self.mu_c_unscaled.shape
        assert var.shape == self.var_c_unscaled.shape
        assert phi.shape == self.phi_c_unscaled.shape
        self.mu_c_unscaled = torch.nn.Parameter(mu)
        self.var_c_unscaled = torch.nn.Parameter(var)
        self.phi_c_unscaled = torch.tensor(phi)

    def forward(self):
        mu_c = self.mu_c_unscaled
        var_c = F.softplus(self.var_c_unscaled)
        phi_c = torch.softmax(self.phi_c_unscaled, dim=0)
        return mu_c, var_c, phi_c


class _VaDER(nn.Module):
    """

    Parameters
    ----------
    n_steps :
    d_input :
    n_clusters :
    d_rnn_hidden :
    d_mu_stddev :
    eps :
    alpha : float, default=1.0
        Weight of the latent loss.
        The final loss = `alpha`*latent loss + reconstruction loss


    Attributes
    ----------

    """

    def __init__(self, n_steps, d_input, n_clusters, d_rnn_hidden, d_mu_stddev, eps=1e-9, alpha=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.d_input = d_input
        self.n_clusters = n_clusters
        self.d_rnn_hidden = d_rnn_hidden
        self.d_mu_stddev = d_mu_stddev
        self.eps = eps
        self.alpha = alpha

        # building model components
        self.implicit_imputation_layer = ImplicitImputation(d_input)
        self.encoder = PeepholeLSTMCell(d_input, d_rnn_hidden)
        self.decoder = PeepholeLSTMCell(d_input, d_rnn_hidden)
        self.ae_encode_layers = nn.Sequential(
            nn.Linear(d_rnn_hidden, d_rnn_hidden),
            nn.Softplus()
        )
        self.ae_decode_layers = nn.Sequential(
            nn.Linear(d_mu_stddev, d_rnn_hidden),
            nn.Softplus()
        )
        self.mu_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)  # layer for mean
        self.stddev_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)  # layer for standard variance
        self.rnn_transform_layer = nn.Linear(d_rnn_hidden, d_input)
        self.gmm_layer = GMMLayer(d_mu_stddev, n_clusters)

    def z_sampling(self, mu_tilde, stddev_tilde):
        noise = mu_tilde.data.new(mu_tilde.size()).normal_()
        z = torch.add(mu_tilde, torch.exp(0.5 * stddev_tilde) * noise)
        return z

    def encode(self, X, missing_mask):
        batch_size = X.size(0)

        X_imputed = self.implicit_imputation_layer(X, missing_mask)

        hidden_state = torch.zeros((batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device)
        cell_state = torch.zeros((batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device)
        # cell_state_collector = torch.empty((batch_size, self.n_steps, self.d_rnn_hidden),
        #                                    dtype=X.dtype, device=X.device)
        for i in range(self.n_steps):
            x = X_imputed[:, i, :]
            hidden_state, cell_state = self.encoder(x, (hidden_state, cell_state))
            # cell_state_collector[:, i, :] = cell_state

        cell_state_collector = self.ae_encode_layers(cell_state)
        mu_tilde = self.mu_layer(cell_state_collector)
        stddev_tilde = self.stddev_layer(cell_state_collector)
        z = self.z_sampling(mu_tilde, stddev_tilde)
        return z, mu_tilde, stddev_tilde

    def decode(self, z):
        hidden_state = z
        hidden_state = self.ae_decode_layers(hidden_state)

        cell_state = torch.zeros(hidden_state.size(), dtype=z.dtype, device=z.device)
        inputs = torch.zeros((z.size(0), self.n_steps, self.d_input), dtype=z.dtype, device=z.device)

        hidden_state_collector = torch.empty((z.size(0), self.n_steps, self.d_rnn_hidden),
                                             dtype=z.dtype, device=z.device)
        for i in range(self.n_steps):
            x = inputs[:, i, :]
            hidden_state, cell_state = self.decoder(x, (hidden_state, cell_state))
            hidden_state_collector[:, i, :] = hidden_state

        reconstruction = self.rnn_transform_layer(hidden_state_collector)
        return reconstruction

    def get_results(self, X, missing_mask):
        z, mu_tilde, stddev_tilde = self.encode(X, missing_mask)
        X_reconstructed = self.decode(z)
        mu_c, var_c, phi_c = self.gmm_layer()
        return X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde

    def cluster(self, inputs):
        X, missing_mask = inputs['X'], inputs['missing_mask']
        X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde = self.get_results(X, missing_mask)

        def func_to_apply(mu_t_, mu_, stddev_, phi_):
            # the covariance matrix is diagonal, so we can just take the product
            return np.log(self.eps + phi_) + \
                   np.log(self.eps + multivariate_normal.pdf(mu_t_, mean=mu_, cov=np.diag(stddev_)))

        mu_tilde = mu_tilde.detach().cpu().numpy()
        mu = mu_c.detach().cpu().numpy()
        var = var_c.detach().cpu().numpy()
        phi = phi_c.detach().cpu().numpy()
        p = np.array([func_to_apply(mu_tilde, mu[i], var[i], phi[i]) for i in np.arange(mu.shape[0])])
        clustering_results = np.argmax(p, axis=0)
        return clustering_results

    def forward(self, inputs, pretrain=False):
        X, missing_mask = inputs['X'], inputs['missing_mask']
        X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde = self.get_results(X, missing_mask)

        # calculate the reconstruction loss
        unscaled_reconstruction_loss = cal_mse(X_reconstructed, X, missing_mask)
        reconstruction_loss = unscaled_reconstruction_loss * self.n_steps * self.d_input / missing_mask.sum()
        if pretrain:
            results = {
                'loss': reconstruction_loss,
                'z': z
            }
            return results

        # calculate the latent loss
        var_tilde = torch.exp(stddev_tilde)
        stddev_c = torch.log(var_c + self.eps)
        log_2pi = torch.log(torch.FloatTensor([2 * torch.pi]))
        log_phi_c = torch.log(phi_c + self.eps)

        batch_size = z.shape[0]

        ii, jj = torch.meshgrid(
            torch.arange(self.n_clusters, dtype=torch.int64, device=X.device),
            torch.arange(batch_size, dtype=torch.int64, device=X.device)
        )
        ii = ii.flatten()
        jj = jj.flatten()

        lsc_b = stddev_c.index_select(dim=0, index=ii)
        mc_b = mu_c.index_select(dim=0, index=ii)
        sc_b = var_c.index_select(dim=0, index=ii)
        z_b = z.index_select(dim=0, index=jj)
        log_pdf_z = - 0.5 * (lsc_b + log_2pi + torch.square(z_b - mc_b) / sc_b)
        log_pdf_z = log_pdf_z.reshape([batch_size, self.n_clusters, self.d_mu_stddev])

        log_p = log_phi_c + log_pdf_z.sum(dim=2)
        lse_p = log_p.logsumexp(dim=1, keepdim=True)
        log_gamma_c = log_p - lse_p
        gamma_c = torch.exp(log_gamma_c)

        term1 = torch.log(var_c + self.eps)
        st_b = var_tilde.index_select(dim=0, index=jj)
        sc_b = var_c.index_select(dim=0, index=ii)
        term2 = torch.reshape(st_b / (sc_b + self.eps), [batch_size, self.n_clusters, self.d_mu_stddev])
        mt_b = mu_tilde.index_select(dim=0, index=jj)
        mc_b = mu_c.index_select(dim=0, index=ii)
        term3 = torch.reshape(
            torch.square(mt_b - mc_b) / (sc_b + self.eps),
            [batch_size, self.n_clusters, self.d_mu_stddev]
        )

        latent_loss1 = 0.5 * torch.sum(gamma_c * torch.sum(term1 + term2 + term3, dim=2), dim=1)
        latent_loss2 = - torch.sum(gamma_c * (log_phi_c - log_gamma_c), dim=1)
        latent_loss3 = - 0.5 * torch.sum(1 + stddev_tilde, dim=1)

        latent_loss1 = latent_loss1.mean()
        latent_loss2 = latent_loss2.mean()
        latent_loss3 = latent_loss3.mean()
        latent_loss = latent_loss1 + latent_loss2 + latent_loss3

        results = {
            'loss': reconstruction_loss + self.alpha * latent_loss,
            'z': z

        }

        return results


def inverse_softplus(x):
    b = x < 1e2
    x[b] = np.log(np.exp(x[b]) - 1.0 + 1e-9)
    return x


class VaDER(BaseNNClusterer):
    def __init__(self,
                 n_steps,
                 n_features,
                 n_clusters,
                 rnn_hidden_size,
                 d_mu_stddev,
                 learning_rate=1e-3,
                 pretrain_epochs=10,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super().__init__(n_clusters, learning_rate, epochs, patience, batch_size, weight_decay, device)
        self.n_steps = n_steps
        self.n_features = n_features
        self.pretrain_epochs = pretrain_epochs
        self.model = _VaDER(n_steps, n_features, n_clusters, rnn_hidden_size, d_mu_stddev)
        self.model = self.model.to(self.device)
        self._print_model_size()

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
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float('inf')
        self.best_model_dict = None

        # pretrain to initialize parameters of GMM layer
        for epoch in range(self.pretrain_epochs):
            self.model.train()
            for idx, data in enumerate(training_loader):
                inputs = self.assemble_input_data(data)
                self.optimizer.zero_grad()
                results = self.model.forward(inputs, pretrain=True)
                results['loss'].backward()
                self.optimizer.step()
        with torch.no_grad():
            sample_collector = []
            for _ in range(10):  # sampling 10 times
                for idx, data in enumerate(training_loader):
                    inputs = self.assemble_input_data(data)
                    results = self.model.forward(inputs, pretrain=True)
                    sample_collector.append(results['z'])
            samples = torch.cat(sample_collector).cpu().detach().numpy()
            gmm = GaussianMixture(n_components=self.n_clusters, covariance_type="diag", reg_covar=1e-04)
            gmm.fit(samples)
            # get GMM parameters
            phi = np.log(gmm.weights_ + 1e-9)  # inverse softmax
            mu = gmm.means_
            var = inverse_softplus(gmm.covariances_)
            # use trained GMM's parameters to init GMM layer's
            self.model.gmm_layer.set_values(
                torch.from_numpy(mu).to(self.device),
                torch.from_numpy(var).to(self.device),
                torch.from_numpy(phi).to(self.device),
            )
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
                    epoch_val_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self.assemble_input_data(data)
                            results = self.model.forward(inputs)
                            epoch_val_loss_collector.append(results['loss'].item())

                    mean_val_loss = np.mean(epoch_val_loss_collector)
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

    def cluster(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        clustering_results_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self.assemble_input_data(data)
                results = self.model.cluster(inputs)
                clustering_results_collector.append(results)

        clustering_results = np.concatenate(clustering_results_collector)
        return clustering_results
