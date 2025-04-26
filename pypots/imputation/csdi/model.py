"""
The implementation of CSDI for the partially-observed time-series imputation task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _CSDI
from .data import DatasetForCSDI, TestDatasetForCSDI
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...nn.functional import gather_listed_dicts
from ...nn.modules.loss import Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer


class CSDI(BaseNNImputer):
    """The PyTorch implementation of the CSDI model :cite:`tashiro2021csdi`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_layers :
        The number of layers in the CSDI model.

    n_heads :
        The number of heads in the multi-head attention mechanism.

    n_channels :
        The number of residual channels.

    d_time_embedding :
        The dimension number of the time (temporal) embedding.

    d_feature_embedding :
        The dimension number of the feature embedding.

    d_diffusion_embedding :
        The dimension number of the diffusion embedding.

    is_unconditional :
        Whether the model is unconditional or conditional.

    target_strategy :
        The strategy for selecting the target for the diffusion process. It has to be one of ["mix", "random"].

    n_diffusion_steps :
        The number of the diffusion step T in the original paper.

    schedule:
        The schedule for other noise levels. It has to be one of ["quad", "linear"].

    beta_start:
        The minimum noise level.

    beta_end:
        The maximum noise level.

    batch_size :
        The batch size for training and evaluating the model.

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
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
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
        n_layers: int,
        n_heads: int,
        n_channels: int,
        d_time_embedding: int,
        d_feature_embedding: int,
        d_diffusion_embedding: int,
        n_diffusion_steps: int = 50,
        target_strategy: str = "random",
        is_unconditional: bool = False,
        schedule: str = "quad",
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
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
        assert target_strategy in ["mix", "random"]
        assert schedule in ["quad", "linear"]
        self.n_steps = n_steps
        self.target_strategy = target_strategy

        # set up the model
        self.model = _CSDI(
            n_features=n_features,
            n_layers=n_layers,
            n_heads=n_heads,
            n_channels=n_channels,
            d_time_embedding=d_time_embedding,
            d_feature_embedding=d_feature_embedding,
            d_diffusion_embedding=d_diffusion_embedding,
            is_unconditional=is_unconditional,
            n_diffusion_steps=n_diffusion_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X_ori,
            indicating_mask,
            cond_mask,
            observed_tp,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X_ori": X_ori.permute(0, 2, 1),  # ori observed part for model hint
            "indicating_mask": indicating_mask.permute(0, 2, 1),  # for loss calc
            "cond_mask": cond_mask.permute(0, 2, 1),  # for masking X_ori
            "observed_tp": observed_tp,
        }
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        (
            indices,
            X,
            cond_mask,
            observed_tp,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X.permute(0, 2, 1),  # for model input
            "cond_mask": cond_mask.permute(0, 2, 1),  # missing mask
            "observed_tp": observed_tp,
        }
        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        train_dataset = DatasetForCSDI(
            train_set,
            self.target_strategy,
            return_X_ori=False,
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
            val_dataset = DatasetForCSDI(
                val_set,
                self.target_strategy,
                return_X_ori=True,
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
            The number of sampling times for the model to sample from the diffusion process.

        Returns
        -------
        result_dict :
            The dictionary containing the imputation results as key 'imputation' and latent variables if necessary.

        """
        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."

        self.model.eval()  # set the model to evaluation mode

        # Step 1: wrap the input data with classes Dataset and DataLoader
        test_dataset = TestDatasetForCSDI(test_set, return_X_ori=False, file_type=file_type)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Step 2: process the data with the model
        dict_result_collector = []
        for idx, data in enumerate(test_dataloader):
            inputs = self._assemble_input_for_testing(data)
            results = self.model(
                inputs,
                n_sampling_times=n_sampling_times,
            )
            dict_result_collector.append(results)

        # Step 3: output collection and return
        result_dict = gather_listed_dicts(dict_result_collector)

        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> np.ndarray:
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
        results = super().impute(test_set, file_type, n_sampling_times=n_sampling_times)
        return results
