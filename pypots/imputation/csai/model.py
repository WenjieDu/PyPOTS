"""
The implementation of CSAI
"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _BCSAI
from .data import DatasetForCSAI
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...data.saving.h5 import load_dict_from_h5
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class CSAI(BaseNNImputer):
    """The PyTorch implementation of the CSAI model :cite:`qian2023csai`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    rnn_hidden_size :
        The size of the GRU hidden state, also the number of hidden units in the GRU cell.

    imputation_weight :
        The weight assigned to the reconstruction loss during training.

    consistency_weight :
        The weight assigned to the consistency loss during training.

    removal_percent :
        The percentage of data to be removed during training for imputation tasks.

    increase_factor :
        A scaling factor used to adjust the amount of missing data during training.

    step_channels :
        The number of channels for each step in the sequence.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    train_loss_func :
        The customized loss function designed by users for training the model.
        If not given, will use the default loss as claimed in the original paper.

    val_metric_func:
        The customized metric function designed by users for validating the model.
        If not given, will use the default MSE metric.

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

    Notes
    -----
    CSAI (Consistent Sequential Imputation) is a bidirectional model designed for time-series imputation.
    It employs a forward and backward GRU network to handle missing data, using consistency and reconstruction losses
    to improve accuracy. The model supports various training configurations, such as interval computations,
    early-stopping, and multiple devices for training. Results can be saved based on the specified saving strategy,
    and tensorboard files are generated for tracking the model's performance over time.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        imputation_weight: float,
        consistency_weight: float,
        removal_percent: int,
        increase_factor: float,
        step_channels: int,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        train_loss_func: Optional[dict] = None,
        val_metric_func: Optional[dict] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Union[str, torch.device, list, None] = None,
        saving_path: str = None,
        model_saving_strategy: Union[str, None] = "best",
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

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.removal_percent = removal_percent
        self.increase_factor = increase_factor
        self.step_channels = step_channels

        # Initialise model
        self.model = _BCSAI(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.step_channels,
            self.consistency_weight,
            self.imputation_weight,
        )

        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # extract data
        sample = data["sample"]

        (indices, X, missing_mask, deltas, last_obs, back_X, back_missing_mask, back_deltas, back_last_obs) = (
            self._send_data_to_given_device(sample)
        )

        # assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
                "last_obs": last_obs,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
                "last_obs": back_last_obs,
            },
            "intervals": self.intervals,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        # extract data
        sample = data["sample"]
        (
            indices,
            X,
            missing_mask,
            deltas,
            last_obs,
            back_X,
            back_missing_mask,
            back_deltas,
            back_last_obs,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(sample)

        # assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
                "last_obs": last_obs,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
                "last_obs": back_last_obs,
            },
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
            "intervals": self.intervals,
        }
        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)

    def fit(
        self,
        train_set,
        val_set=None,
        file_type: str = "hdf5",
    ) -> None:

        if isinstance(train_set, str):
            logger.warning(
                "CSAI does not support lazy loading because intervals need to be calculated ahead. "
                "Hence the whole train set will be loaded into memory."
            )
            train_set = load_dict_from_h5(train_set)

        training_set = DatasetForCSAI(
            train_set,
            False,
            False,
            file_type,
            self.removal_percent,
            self.increase_factor,
        )
        self.intervals = training_set.intervals
        self.replacement_probabilities = training_set.replacement_probabilities

        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if isinstance(val_set, str):
                logger.warning(
                    "CSAI does not support lazy loading because intervals need to be calculated ahead. "
                    "Hence the whole val set will be loaded into memory."
                )
                val_set = load_dict_from_h5(val_set)

            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            validating_set = DatasetForCSAI(
                val_set,
                True,
                False,
                file_type,
                self.removal_percent,
                self.increase_factor,
                self.replacement_probabilities,
            )
            val_loader = DataLoader(
                validating_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # train the model
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:

        self.model.eval()

        if isinstance(test_set, str):
            logger.warning(
                "CSAI does not support lazy loading because intervals need to be calculated ahead. "
                "Hence the whole test set will be loaded into memory."
            )
            test_set = load_dict_from_h5(test_set)
        testing_set = DatasetForCSAI(
            test_set,
            True,
            False,
            file_type,
            self.removal_percent,
            self.increase_factor,
            self.replacement_probabilities,
        )

        test_loader = DataLoader(
            testing_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        imputation_collector = []
        x_ori_collector = []
        indicating_mask_collector = []

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)
                x_ori_collector.append(inputs["X_ori"])
                indicating_mask_collector.append(inputs["indicating_mask"])

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
            "X_ori": torch.cat(x_ori_collector).cpu().detach().numpy(),
            "indicating_mask": torch.cat(indicating_mask_collector).cpu().detach().numpy(),
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

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]
