"""
The implementation of VaDER for the partially-observed time-series clustering task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
from typing import Union, Optional

import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from .core import inverse_softplus, _VaDER
from .data import DatasetForVaDER
from ..base import BaseNNClusterer
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger

try:
    import nni
except ImportError:
    pass


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
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.
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
        patience: Optional[int] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            n_clusters=n_clusters,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            train_loss_func=None,
            val_metric_func=None,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        assert pretrain_epochs > 0, f"pretrain_epochs must be a positive integer, but got {pretrain_epochs}"

        self.n_steps = n_steps
        self.n_features = n_features
        self.pretrain_epochs = pretrain_epochs

        # set up the model
        self.model = _VaDER(n_steps, n_features, n_clusters, rnn_hidden_size, d_mu_stddev)
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
                    self._save_log_into_tb_file(pretraining_step, "pretraining", results)

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
                    logger.error(f"❌ Exception: {e}")
                    logger.warning("‼️ Met with ValueError, double `reg_covar` to re-train the GMM model.")

                    flag -= 1
                    if flag == -5:
                        logger.error(
                            f"❌ Doubled `reg_covar` for 4 times, its current value is {reg_covar}, but still failed.\n"
                            f"Now quit to let you check your model training.\n"
                            "Please raise an issue https://github.com/WenjieDu/PyPOTS/issues if you have questions."
                        )
                        raise RuntimeError
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
                self.model.module.backbone.gmm_layer.set_values(
                    torch.from_numpy(mu).to(device),
                    torch.from_numpy(var).to(device),
                    torch.from_numpy(phi).to(device),
                )
            else:
                self.model.backbone.gmm_layer.set_values(
                    torch.from_numpy(mu).to(device),
                    torch.from_numpy(var).to(device),
                    torch.from_numpy(phi).to(device),
                )

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
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
                            epoch_val_loss_collector.append(results["loss"].sum().item())

                    mean_val_loss = np.mean(epoch_val_loss_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss ({self.train_loss_func_name}): {mean_train_loss:.4f}, "
                        f"validation {self.val_metric_func_name}: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(
                        f"Epoch {epoch:03d} - training loss ({self.train_loss_func_name}): {mean_train_loss:.4f}"
                    )
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss:.4f}",
                )

                if os.getenv("enable_tuning", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForVaDER(train_set, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_loader = None
        if val_set is not None:
            val_set = DatasetForVaDER(val_set, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        return_latent_vars: bool = False,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set : dict or str
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        return_latent_vars : bool
            Whether to return the latent variables in VaDER, e.g. mu and phi, etc.

        Returns
        -------
        file_type :
            The dictionary containing the clustering results and latent variables if necessary.

        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForVaDER(test_set, return_y=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        mu_tilde_collector = []
        stddev_tilde_collector = []
        mu_collector = []
        var_collector = []
        phi_collector = []
        z_collector = []
        imputation_latent_collector = []
        clustering_results_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs)

                mu_tilde = results["mu_tilde"].cpu().numpy()
                mu_tilde_collector.append(mu_tilde)
                mu = results["mu"].cpu().numpy()
                mu_collector.append(mu)
                var = results["var"].cpu().numpy()
                var_collector.append(var)
                phi = results["phi"].cpu().numpy()
                phi_collector.append(phi)

                def func_to_apply(
                    mu_t_: np.ndarray,
                    mu_: np.ndarray,
                    stddev_: np.ndarray,
                    phi_: np.ndarray,
                ) -> np.ndarray:
                    # the covariance matrix is diagonal, so we can just take the product
                    return np.log(1e-9 + phi_) + np.log(
                        1e-9 + multivariate_normal.pdf(mu_t_, mean=mu_, cov=np.diag(stddev_))
                    )

                p = np.array([func_to_apply(mu_tilde, mu[i], var[i], phi[i]) for i in np.arange(mu.shape[0])])
                clustering_results = np.argmax(p, axis=0)
                clustering_results_collector.append(clustering_results)

                if return_latent_vars:
                    stddev_tilde = results["stddev_tilde"].cpu().numpy()
                    stddev_tilde_collector.append(stddev_tilde)
                    z = results["z"].cpu().numpy()
                    z_collector.append(z)
                    imputation_latent = results["imputation_latent"].cpu().numpy()
                    imputation_latent_collector.append(imputation_latent)

        clustering = np.concatenate(clustering_results_collector)
        result_dict = {
            "clustering": clustering,
        }

        if return_latent_vars:
            latent_var_collector = {
                "mu_tilde": np.concatenate(mu_tilde_collector),
                "stddev_tilde": np.concatenate(stddev_tilde_collector),
                "mu": np.concatenate(mu_collector),
                "var": np.concatenate(var_collector),
                "phi": np.concatenate(phi_collector),
                "z": np.concatenate(z_collector),
                "imputation_latent": np.concatenate(imputation_latent_collector),
            }
            result_dict["latent_vars"] = latent_var_collector

        return result_dict

    def cluster(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> Union[np.ndarray]:
        """Cluster the input with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like,
            Clustering results.

        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["clustering"]
