"""
Torch implementation of model VaDER.

Refer to paper :cite:`dejong2019VaDER`.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Tuple, Union, Optional

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
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mse


class ImplicitImputation(nn.Module):
    def __init__(self, d_input: int):
        super().__init__()
        self.projection_layer = nn.Linear(d_input, d_input, bias=False)

    def forward(self, X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        imputation = self.projection_layer(X)
        imputed_X = X * missing_mask + imputation * (1 - X)
        return imputed_X


class PeepholeLSTMCell(nn.LSTMCell):
    """
    Notes
    -----
    This implementation is adapted from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81

    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias)
        self.weight_ch = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ch = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter("bias_ch", None)
        self.register_buffer("wc_blank", torch.zeros(hidden_size))
        self.reset_parameters()

    def forward(
        self,
        X: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            zeros = torch.zeros(
                X.size(0), self.hidden_size, dtype=X.dtype, device=X.device
            )
            hx = (zeros, zeros)

        h, c = hx

        wx = F.linear(X, self.weight_ih, self.bias_ih)
        wh = F.linear(h, self.weight_hh, self.bias_hh)
        wc = F.linear(c, self.weight_ch, self.bias_ch)

        wxhc = (
            wx
            + wh
            + torch.cat(
                (
                    wc[:, : 2 * self.hidden_size],
                    Variable(self.wc_blank).expand_as(h),
                    wc[:, 2 * self.hidden_size :],
                ),
                dim=1,
            )
        )

        i = torch.sigmoid(wxhc[:, : self.hidden_size])
        f = torch.sigmoid(wxhc[:, self.hidden_size : 2 * self.hidden_size])
        g = torch.tanh(wxhc[:, 2 * self.hidden_size : 3 * self.hidden_size])
        o = torch.sigmoid(wxhc[:, 3 * self.hidden_size :])

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class GMMLayer(nn.Module):
    def __init__(self, d_hidden: int, n_clusters: int):
        super().__init__()
        self.mu_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.var_c_unscaled = Parameter(torch.Tensor(n_clusters, d_hidden))
        self.phi_c_unscaled = torch.Tensor(n_clusters)

    def set_values(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        phi: torch.Tensor,
    ) -> None:
        assert mu.shape == self.mu_c_unscaled.shape
        assert var.shape == self.var_c_unscaled.shape
        assert phi.shape == self.phi_c_unscaled.shape
        self.mu_c_unscaled = torch.nn.Parameter(mu)
        self.var_c_unscaled = torch.nn.Parameter(var)
        self.phi_c_unscaled = phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def __init__(
        self,
        n_steps: int,
        d_input: int,
        n_clusters: int,
        d_rnn_hidden: int,
        d_mu_stddev: int,
        eps: float = 1e-9,
        alpha: float = 1.0,
    ):
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
            nn.Linear(d_rnn_hidden, d_rnn_hidden), nn.Softplus()
        )
        self.ae_decode_layers = nn.Sequential(
            nn.Linear(d_mu_stddev, d_rnn_hidden), nn.Softplus()
        )
        self.mu_layer = nn.Linear(d_rnn_hidden, d_mu_stddev)  # layer for mean
        self.stddev_layer = nn.Linear(
            d_rnn_hidden, d_mu_stddev
        )  # layer for standard variance
        self.rnn_transform_layer = nn.Linear(d_rnn_hidden, d_input)
        self.gmm_layer = GMMLayer(d_mu_stddev, n_clusters)

    @staticmethod
    def z_sampling(
        mu_tilde: torch.Tensor,
        stddev_tilde: torch.Tensor,
    ) -> torch.Tensor:
        noise = mu_tilde.data.new(mu_tilde.size()).normal_()
        z = torch.add(mu_tilde, torch.exp(0.5 * stddev_tilde) * noise)
        return z

    def encode(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = X.size(0)

        X_imputed = self.implicit_imputation_layer(X, missing_mask)

        hidden_state = torch.zeros(
            (batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device
        )
        cell_state = torch.zeros(
            (batch_size, self.d_rnn_hidden), dtype=X.dtype, device=X.device
        )
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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden_state = z
        hidden_state = self.ae_decode_layers(hidden_state)

        cell_state = torch.zeros(hidden_state.size(), dtype=z.dtype, device=z.device)
        inputs = torch.zeros(
            (z.size(0), self.n_steps, self.d_input), dtype=z.dtype, device=z.device
        )

        hidden_state_collector = torch.empty(
            (z.size(0), self.n_steps, self.d_rnn_hidden), dtype=z.dtype, device=z.device
        )
        for i in range(self.n_steps):
            x = inputs[:, i, :]
            hidden_state, cell_state = self.decoder(x, (hidden_state, cell_state))
            hidden_state_collector[:, i, :] = hidden_state

        reconstruction = self.rnn_transform_layer(hidden_state_collector)
        return reconstruction

    def get_results(
        self, X: torch.Tensor, missing_mask: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        z, mu_tilde, stddev_tilde = self.encode(X, missing_mask)
        X_reconstructed = self.decode(z)
        mu_c, var_c, phi_c = self.gmm_layer()
        return X_reconstructed, mu_c, var_c, phi_c, z, mu_tilde, stddev_tilde

    def cluster(self, inputs: dict) -> np.ndarray:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        (
            X_reconstructed,
            mu_c,
            var_c,
            phi_c,
            z,
            mu_tilde,
            stddev_tilde,
        ) = self.get_results(X, missing_mask)

        def func_to_apply(
            mu_t_: np.ndarray, mu_: np.ndarray, stddev_: np.ndarray, phi_: np.ndarray
        ) -> np.ndarray:
            # the covariance matrix is diagonal, so we can just take the product
            return np.log(self.eps + phi_) + np.log(
                self.eps
                + multivariate_normal.pdf(mu_t_, mean=mu_, cov=np.diag(stddev_))
            )

        mu_tilde = mu_tilde.detach().cpu().numpy()
        mu = mu_c.detach().cpu().numpy()
        var = var_c.detach().cpu().numpy()
        phi = phi_c.detach().cpu().numpy()
        p = np.array(
            [
                func_to_apply(mu_tilde, mu[i], var[i], phi[i])
                for i in np.arange(mu.shape[0])
            ]
        )
        clustering_results = np.argmax(p, axis=0)
        return clustering_results

    def forward(self, inputs: dict, pretrain: bool = False) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        (
            X_reconstructed,
            mu_c,
            var_c,
            phi_c,
            z,
            mu_tilde,
            stddev_tilde,
        ) = self.get_results(X, missing_mask)

        device = X.device

        # calculate the reconstruction loss
        unscaled_reconstruction_loss = cal_mse(X_reconstructed, X, missing_mask)
        reconstruction_loss = (
            unscaled_reconstruction_loss
            * self.n_steps
            * self.d_input
            / missing_mask.sum()
        )
        if pretrain:
            results = {"loss": reconstruction_loss, "z": z}
            return results

        # calculate the latent loss
        var_tilde = torch.exp(stddev_tilde)
        stddev_c = torch.log(var_c + self.eps)
        log_2pi = torch.log(torch.tensor([2 * torch.pi], device=device))
        log_phi_c = torch.log(phi_c + self.eps)

        batch_size = z.shape[0]

        ii, jj = torch.meshgrid(
            torch.arange(self.n_clusters, dtype=torch.int64, device=device),
            torch.arange(batch_size, dtype=torch.int64, device=device),
            indexing="ij",
        )
        ii = ii.flatten()
        jj = jj.flatten()

        lsc_b = stddev_c.index_select(dim=0, index=ii)
        mc_b = mu_c.index_select(dim=0, index=ii)
        sc_b = var_c.index_select(dim=0, index=ii)
        z_b = z.index_select(dim=0, index=jj)
        log_pdf_z = -0.5 * (lsc_b + log_2pi + torch.square(z_b - mc_b) / sc_b)
        log_pdf_z = log_pdf_z.reshape([batch_size, self.n_clusters, self.d_mu_stddev])

        log_p = log_phi_c + log_pdf_z.sum(dim=2)
        lse_p = log_p.logsumexp(dim=1, keepdim=True)
        log_gamma_c = log_p - lse_p
        gamma_c = torch.exp(log_gamma_c)

        term1 = torch.log(var_c + self.eps)
        st_b = var_tilde.index_select(dim=0, index=jj)
        sc_b = var_c.index_select(dim=0, index=ii)
        term2 = torch.reshape(
            st_b / (sc_b + self.eps), [batch_size, self.n_clusters, self.d_mu_stddev]
        )
        mt_b = mu_tilde.index_select(dim=0, index=jj)
        mc_b = mu_c.index_select(dim=0, index=ii)
        term3 = torch.reshape(
            torch.square(mt_b - mc_b) / (sc_b + self.eps),
            [batch_size, self.n_clusters, self.d_mu_stddev],
        )

        latent_loss1 = 0.5 * torch.sum(
            gamma_c * torch.sum(term1 + term2 + term3, dim=2), dim=1
        )
        latent_loss2 = -torch.sum(gamma_c * (log_phi_c - log_gamma_c), dim=1)
        latent_loss3 = -0.5 * torch.sum(1 + stddev_tilde, dim=1)

        latent_loss1 = latent_loss1.mean()
        latent_loss2 = latent_loss2.mean()
        latent_loss3 = latent_loss3.mean()
        latent_loss = latent_loss1 + latent_loss2 + latent_loss3

        results = {"loss": reconstruction_loss + self.alpha * latent_loss, "z": z}

        return results


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    b = x < 1e2
    x[b] = np.log(np.exp(x[b]) - 1.0 + 1e-9)
    return x


class VaDER(BaseNNClusterer):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_clusters: int,
        rnn_hidden_size: int,
        d_mu_stddev: int,
        batch_size: int = 32,
        epochs: int = 100,
        pretrain_epochs: int = 10,
        patience: int = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            n_clusters,
            batch_size,
            epochs,
            patience,
            learning_rate,
            weight_decay,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )
        self.n_steps = n_steps
        self.n_features = n_features
        self.pretrain_epochs = pretrain_epochs
        self.model = _VaDER(
            n_steps, n_features, n_clusters, rnn_hidden_size, d_mu_stddev
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble the given data into a dictionary for training input.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model training.
        """

        # fetch data
        indices, X, _, missing_mask, _, _ = map(lambda x: x.to(self.device), data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        """Assemble the given data into a dictionary for validating input.

        Notes
        -----
        The validating data assembling processing is the same as training data assembling.


        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model validating.
        """
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """Assemble the given data into a dictionary for testing input.

        Notes
        -----
        The testing data assembling processing is the same as training data assembling.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model testing.
        """
        return self._assemble_input_for_validating(data)

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        # pretrain to initialize parameters of GMM layer
        pretraining_step = 0
        for epoch in range(self.pretrain_epochs):
            self.model.train()
            for idx, data in enumerate(training_loader):
                pretraining_step += 1
                inputs = self._assemble_input_for_training(data)
                self.optimizer.zero_grad()
                results = self.model.forward(inputs, pretrain=True)
                results["loss"].backward()
                self.optimizer.step()

                # save pre-training loss logs into the tensorboard file for every step if in need
                if self.summary_writer is not None:
                    self.save_log_into_tb_file(pretraining_step, "pretraining", results)

        with torch.no_grad():
            sample_collector = []
            for _ in range(10):  # sampling 10 times
                for idx, data in enumerate(training_loader):
                    inputs = self._assemble_input_for_validating(data)
                    results = self.model.forward(inputs, pretrain=True)
                    sample_collector.append(results["z"])
            samples = torch.cat(sample_collector).cpu().detach().numpy()

            # leverage the below loop to automatically fix the exception ValueError raised by gmm.fit()
            flag = 0
            reg_covar = 1e-04
            while flag <= 0:
                try:
                    gmm = GaussianMixture(
                        n_components=self.n_clusters,
                        covariance_type="diag",
                        reg_covar=reg_covar,
                        # reg_covar is set as 1e-04 in the official implementation, but may cause ValueError: Fitting
                        # the mixture model failed because some components have ill-defined empirical covariance
                        # (for instance caused by singleton or collapsed samples). Try to decrease the number
                        # of components, or increase reg_covar.
                    )
                    gmm.fit(samples)
                    flag = 1
                except ValueError as e:
                    logger.error(e)
                    logger.warning(
                        "Met with ValueError, double `reg_covar` to re-train the GMM model."
                    )

                    flag -= 1
                    if flag == -5:
                        logger.error(
                            f"Doubled `reg_covar` for 4 times, whose current value is {reg_covar}, but still failed.\n"
                            "Now quit to let you check your model training.\n"
                            "Please raise an issue https://github.com/WenjieDu/PyPOTS/issues if you have questions."
                        )
                        exit()
                    else:
                        reg_covar *= 2

                    continue

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
            training_step = 0
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)
                    self.optimizer.zero_grad()
                    results = self.model.forward(inputs)
                    results["loss"].backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self.save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    epoch_val_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model.forward(inputs)
                            epoch_val_loss_collector.append(results["loss"].item())

                    mean_val_loss = np.mean(epoch_val_loss_collector)

                    # save validating loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "loss": mean_val_loss,
                        }
                        self.save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"epoch {epoch}: "
                        f"training loss {mean_train_loss:.4f}, "
                        f"validating loss {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"epoch {epoch}: training loss {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                    # save the model if necessary
                    self.auto_save_model_if_necessary(
                        training_finished=False,
                        saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss}",
                    )
                else:
                    self.patience -= 1
                    if self.patience == 0:
                        logger.info(
                            "Exceeded the training patience. Terminating the training procedure..."
                        )
                        break
        except Exception as e:
            logger.info(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not get trained. Please try fit() again."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. "
                    "Model will load the best parameters so far for testing. "
                    "If you don't want it, please try fit() again."
                )

        if np.equal(self.best_loss, float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info("Finished training.")

    def fit(
        self,
        train_set: Union[dict, str],
        file_type: str = "h5py",
    ) -> None:
        """Train the cluster.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type : str, default = "h5py"
            The type of the given file if train_set is a path string.

        Returns
        -------
        self : object,
            Trained classifier.
        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGRUD(
            train_set, return_labels=False, file_type=file_type
        )
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Step 2: train the model and freeze it
        self._train_model(training_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self.auto_save_model_if_necessary(training_finished=True)

    def cluster(self, X: Union[dict, str], file_type: str = "h5py") -> np.ndarray:
        """Cluster the input with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py"
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Clustering results.
        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(X, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        clustering_results_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.cluster(inputs)
                clustering_results_collector.append(results)

        clustering_results = np.concatenate(clustering_results_collector)
        return clustering_results
