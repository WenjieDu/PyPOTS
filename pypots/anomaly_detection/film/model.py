"""
The implementation of FiLM for the partially-observed time-series anomaly detection task.
"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...imputation.film.core import _FiLM
from ...imputation.saits.data import DatasetForSAITS
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class FiLM(BaseNNDetector):
    """The PyTorch implementation of the FiLM model for the anomaly detection task.
    FiLM is originally proposed by Zhou et al. in :cite:`zhou2022film`.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    anomaly_rate : float
        The estimated anomaly rate in the dataset, within the range (0, 1). Used for thresholding.

    window_size : list
        A list including the window sizes for the HiPPO projection layers.

    multiscale : list
        A list including the multiscale factors for the HiPPO projection layers.

    modes1 : int, optional
        The number of Fourier modes used. Default is 32.

    dropout : float, optional
        The dropout ratio for the HiPPO projection layers. Default is 0.5.

    mode_type : int, optional
        The mode type of the SpectralConv1d layers. Must be one of {0, 1, 2}.

    d_model : int, optional
        The dimension of the model. Default is 128.

    ORT_weight : float, optional
        The weight for the ORT loss. Default is 1.

    MIT_weight : float, optional
        The weight for the MIT loss. Default is 1.

    batch_size : int, optional
        The number of samples per batch during training and evaluation.

    epochs : int, optional
        Total number of training epochs.

    patience : int, optional
        Number of epochs to wait for improvement before triggering early stopping. Disabled if None.

    training_loss : Criterion or type, optional
        Loss function used during training. Defaults to MAE.

    validation_metric : Criterion or type, optional
        Metric used during validation. Defaults to MSE.

    optimizer : Optimizer or type, optional
        Optimizer used for training. Defaults to custom Adam optimizer.

    num_workers : int, optional
        Number of subprocesses used for data loading.

    device : str, torch.device, or list, optional
        Device(s) used for model training and inference.

    saving_path : str, optional
        Path to save model checkpoints and training logs.

    model_saving_strategy : str or None, optional
        Strategy to save models: one of {None, "best", "better", "all"}.

    verbose : bool, optional
        Whether to print training logs during execution.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        window_size: list,
        multiscale: list,
        modes1: int = 32,
        dropout: float = 0.5,
        mode_type: int = 0,
        d_model: int = 128,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            anomaly_rate=anomaly_rate,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        assert mode_type in [0, 1, 2], "mode_type should be 0, 1, or 2."

        self.n_steps = n_steps
        self.n_features = n_features
        self.window_size = window_size
        self.multiscale = multiscale
        self.modes1 = modes1
        self.dropout = dropout
        self.mode_type = mode_type
        self.d_model = d_model
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # Set up the model
        self.model = _FiLM(
            n_steps=self.n_steps,
            n_features=self.n_features,
            window_size=self.window_size,
            multiscale=self.multiscale,
            modes1=self.modes1,
            ratio=self.dropout,
            mode_type=self.mode_type,
            d_model=self.d_model,
            ORT_weight=self.ORT_weight,
            MIT_weight=self.MIT_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # Set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """
        Prepares input batch for training during anomaly detection.
        """
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        return {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        """
        Prepares input batch for validation during anomaly detection.
        """
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """
        Prepares input batch for inference (testing) during anomaly detection.
        """
        indices, X, missing_mask = self._send_data_to_given_device(data)

        return {
            "X": X,
            "missing_mask": missing_mask,
        }

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """
        Trains the FiLM model on the given dataset for anomaly detection.
        """
        self.train_set = train_set

        # Step 1: Wrap training set with Dataset and DataLoader
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Step 2: Wrap validation set (if provided)
        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_dataset = DatasetForSAITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 3: Train the model
        self._train_model(train_dataloader, val_dataloader)

        # Step 4: Restore the best model from training
        self.model.load_state_dict(self.best_model_dict)

        # Step 5: Save the model if required
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
