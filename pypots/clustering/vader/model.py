"""
The implementation of VaDER for the partially-observed time-series clustering task.

Refer to the paper "Jong, J.D., Emon, M.A., Wu, P., Karki, R., Sood, M., Godard, P., Ahmad, A., Vrooman, H.A.,
Hofmann-Apitius, M., & Fröhlich, H. (2019).
Deep learning for clustering of multivariate clinical patient trajectories with missing values. GigaScience."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from .data import DatasetForVaDER
from .modules import (
    inverse_softplus,
    GMMLayer,
    PeepholeLSTMCell,
    ImplicitImputation,
)
from ..base import BaseNNClusterer
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from ...utils.metrics import cal_mse


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
    alpha :
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

    def forward(
        self,
        inputs: dict,
        pretrain: bool = False,
        training: bool = True,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        device = X.device

        (
            X_reconstructed,
            mu_c,
            var_c,
            phi_c,
            z,
            mu_tilde,
            stddev_tilde,
        ) = self.get_results(X, missing_mask)

        if not training and not pretrain:

            results = {
                "mu_tilde": mu_tilde,
                "mu": mu_c,
                "var": var_c,
                "phi": phi_c,
            }
            # if only run clustering, then no need to calculate loss
            return results

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

        results = {
            "loss": reconstruction_loss + self.alpha * latent_loss,
            "z": z,
        }

        return results


class VaDER(BaseNNClusterer):
    """The PyTorch implementation of the VaDER model :cite:`dejong2019VaDER`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_clusters :
        The number of clusters in the clustering task.

    rnn_hidden_size :
        The size of the RNN hidden state, also the number of hidden units in the RNN cell.

    d_mu_stddev :
        The dimension of the mean and standard deviation of the Gaussian distribution.

    batch_size :
        The batch size for training and evaluating the model.

    pretrain_epochs :
        The number of epochs for pretraining the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.
    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on.
        If not given, will try to use CUDA devices first (will use the GPU with device number 0 only by default),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    Attributes
    ----------
    model : :class:`torch.nn.Module`
        The underlying VaDER model.

    optimizer : :class:`pypots.optim.Optimizer`
        The optimizer for model training.

    """

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
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            n_clusters,
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )

        assert (
            pretrain_epochs > 0
        ), f"pretrain_epochs must be a positive integer, but got {pretrain_epochs}"

        self.n_steps = n_steps
        self.n_features = n_features
        self.pretrain_epochs = pretrain_epochs

        # set up the model
        self.model = _VaDER(
            n_steps, n_features, n_clusters, rnn_hidden_size, d_mu_stddev
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:

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
                results["loss"].sum().backward()
                self.optimizer.step()

                # save pre-training loss logs into the tensorboard file for every step if in need
                if self.summary_writer is not None:
                    self._save_log_into_tb_file(
                        pretraining_step, "pretraining", results
                    )

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
            mu = gmm.means_
            var = inverse_softplus(gmm.covariances_)
            phi = np.log(gmm.weights_ + 1e-9)  # inverse softmax
            device = results["z"].device

            # use trained GMM's parameters to init GMM layer's
            if isinstance(self.device, list):  # if using multi-GPU
                self.model.module.gmm_layer.set_values(
                    torch.from_numpy(mu).to(device),
                    torch.from_numpy(var).to(device),
                    torch.from_numpy(phi).to(device),
                )
            else:
                self.model.gmm_layer.set_values(
                    torch.from_numpy(mu).to(device),
                    torch.from_numpy(var).to(device),
                    torch.from_numpy(phi).to(device),
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
                    results["loss"].sum().backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    epoch_val_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model.forward(inputs)
                            epoch_val_loss_collector.append(
                                results["loss"].sum().item()
                            )

                    mean_val_loss = np.mean(epoch_val_loss_collector)

                    # save validating loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

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
                    self._auto_save_model_if_necessary(
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
            logger.error(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
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
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForVaDER(
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
        self._auto_save_model_if_necessary(training_finished=True)

    def cluster(self, X: Union[dict, str], file_type: str = "h5py") -> np.ndarray:
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForVaDER(X, return_labels=False, file_type=file_type)
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
                results = self.model.forward(inputs, training=False)

                mu_tilde = results["mu_tilde"].cpu().numpy()
                mu = results["mu"].cpu().numpy()
                var = results["var"].cpu().numpy()
                phi = results["phi"].cpu().numpy()

                def func_to_apply(
                    mu_t_: np.ndarray,
                    mu_: np.ndarray,
                    stddev_: np.ndarray,
                    phi_: np.ndarray,
                ) -> np.ndarray:
                    # the covariance matrix is diagonal, so we can just take the product
                    return np.log(1e-9 + phi_) + np.log(
                        1e-9
                        + multivariate_normal.pdf(mu_t_, mean=mu_, cov=np.diag(stddev_))
                    )

                p = np.array(
                    [
                        func_to_apply(mu_tilde, mu[i], var[i], phi[i])
                        for i in np.arange(mu.shape[0])
                    ]
                )
                clustering_results = np.argmax(p, axis=0)
                clustering_results_collector.append(clustering_results)

        clustering_results = np.concatenate(clustering_results_collector)
        return clustering_results
