"""
The implementation of GP-VAE for the partially-observed time-series imputation task.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _GPVAE
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...data.dataset.base import BaseDataset
from ...nn.modules.loss import Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer


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
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            training_loss=Criterion,
            validation_metric=Criterion,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
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
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
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

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        train_dataset = BaseDataset(
            train_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_dataset = BaseDataset(
                val_set,
                return_X_ori=True,
                return_X_pred=False,
                return_y=False,
                file_type=file_type,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The test dataset for model to process, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is the time-series data for processing.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include 'X' key.

        file_type :
            The type of the given file if test_set is a path string.

        n_sampling_times:
            The number of sampling times for the model to produce predictions.

        Returns
        -------
        result_dict :
            The dictionary containing the imputation results as key 'imputation' and latent variables if necessary.

        """

        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."
        result_dict = super().predict(test_set, file_type, n_sampling_times=n_sampling_times)
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The test dataset for model to process, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is the time-series data for processing.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include 'X' key.

        file_type :
            The type of the given file if test_set is a path string.

        n_sampling_times:
            The number of sampling times for the model to produce predictions.

        Returns
        -------
        results:
            The imputed data samples.

        """

        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."
        results = super().impute(test_set, file_type, n_sampling_times=n_sampling_times)
        return results
