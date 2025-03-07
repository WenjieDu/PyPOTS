"""
The implementation of GP-VAE for the partially-observed time-series imputation task.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import nni
except ImportError:
    pass


from .core import _GPVAE
from .data import DatasetForGPVAE
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from ...nn.functional import calc_mse


class GPVAE(BaseNNImputer):
    """The PyTorch implementation of the GPVAE model :cite:`fortuin2020gpvae`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    latent_size : int,
        The feature dimension of the latent embedding

    encoder_sizes : tuple,
        The tuple of the network size in encoder

    decoder_sizes : tuple,
        The tuple of the network size in decoder

    beta : float,
        The weight of KL divergence in ELBO.

    M : int,
        The number of Monte Carlo samples for ELBO estimation during training.

    K : int,
        The number of importance weights for IWAE model training loss.

    kernel: str
        The type of kernel function chosen in the Gaussain Process Proir. ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        The scale parameter for a kernel function

    length_scale : float,
        The length scale parameter for a kernel function

    kernel_scales : int,
        The number of different length scales over latent space dimensions

    window_size : int,
        Window size for the inference CNN.

    batch_size : int
        The batch size for training and evaluating the model.

    epochs : int
        The number of epochs for training the model.

    patience : int
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer : pypots.optim.base.Optimizer
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers : int
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : :class:`torch.device` or list
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        latent_size: int,
        encoder_sizes: tuple = (64, 64),
        decoder_sizes: tuple = (64, 64),
        kernel: str = "cauchy",
        beta: float = 0.2,
        M: int = 1,
        K: int = 1,
        sigma: float = 1.0,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        train_loss_func: Optional[dict] = None,
        val_metric_func: Optional[dict] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            train_loss_func=train_loss_func,
            val_metric_func=val_metric_func,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        available_kernel_type = ["cauchy", "diffusion", "rbf", "matern"]
        assert kernel in available_kernel_type, f"kernel should be one of {available_kernel_type}, but got {kernel}"

        self.n_steps = n_steps
        self.n_features = n_features
        self.latent_size = latent_size
        self.kernel = kernel
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.beta = beta
        self.M = M
        self.K = K
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        # set up the model
        self.model = _GPVAE(
            input_dim=self.n_features,
            time_length=self.n_steps,
            latent_dim=self.latent_size,
            kernel=self.kernel,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            beta=self.beta,
            M=self.M,
            K=self.K,
            sigma=self.sigma,
            length_scale=self.length_scale,
            kernel_scales=self.kernel_scales,
            window_size=window_size,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

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
                    # use sum() before backward() in case of multi-gpu training
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
                    imputation_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model.forward(inputs, n_sampling_times=1)
                            imputed_data = results["imputed_data"].mean(axis=1)
                            imputation_mse = (
                                calc_mse(
                                    imputed_data,
                                    inputs["X_ori"],
                                    inputs["indicating_mask"],
                                )
                                .sum()
                                .detach()
                                .item()
                            )
                            imputation_loss_collector.append(imputation_mse)

                    mean_val_loss = np.mean(imputation_loss_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
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

                if os.getenv("ENABLE_HPO", False):
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
        training_set = DatasetForGPVAE(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForGPVAE(val_set, return_X_ori=True, return_y=False, file_type=file_type)
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
        n_sampling_times: int = 1,
    ) -> dict:
        """

        Parameters
        ----------
        test_set : dict or str
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        n_sampling_times:
            The number of sampling times for the model to produce predictions.

        Returns
        -------
        result_dict: dict
            Prediction results in a Python Dictionary for the given samples.
            It should be a dictionary including a key named 'imputation'.

        """
        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."

        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGPVAE(test_set, return_X_ori=False, return_y=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, n_sampling_times=n_sampling_times)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        results_dict = self.predict(test_set, file_type=file_type)
        return results_dict["imputation"]
