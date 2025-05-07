"""
The implementation of TimeMixer for the partially-observed time-series anomaly detection task.
"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...imputation.timemixer.core import _TimeMixer
from ...imputation.saits.data import DatasetForSAITS
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class TimeMixer(BaseNNDetector):
    """The PyTorch implementation of the TimeMixer model for the anomaly detection task.
    TimeMixer is originally proposed by Wang et al. in :cite:`wang2024timemixer`.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    anomaly_rate : float
        The estimated anomaly rate in the dataset, within the range (0, 1). Used for thresholding.

    n_layers : int
        The number of layers in the TimeMixer model.

    d_model : int
        The dimension of the model.

    d_ffn : int
        The dimension of the feed-forward network.

    top_k : int
        The number of top-k frequency amplitudes selected in frequency domain operations.

    dropout : float, optional
        The dropout rate for the model. Default is 0.

    channel_independence : bool, optional
        Whether to use channel independence in the model.

    decomp_method : str, optional
        The decomposition method for the model. Must be one of ['moving_avg', 'dft_decomp'].

    moving_avg : int, optional
        The window size for moving average decomposition.

    downsampling_layers : int, optional
        The number of downsampling layers in the model.

    downsampling_window : int, optional
        The window size for downsampling.

    apply_nonstationary_norm : bool, optional
        Whether to apply non-stationary normalization as described in :cite:`liu2022nonstationary`.

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
        Device(s) used for model training and inference. Supports multi-GPU training.

    saving_path : str, optional
        Path to save model checkpoints and training logs. No saving if None.

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
        n_layers: int,
        d_model: int,
        d_ffn: int,
        top_k: int,
        dropout: float = 0,
        channel_independence: bool = False,
        decomp_method: str = "moving_avg",
        moving_avg: int = 5,
        downsampling_layers: int = 3,
        downsampling_window: int = 2,
        apply_nonstationary_norm: bool = False,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
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

        assert decomp_method in [
            "moving_avg",
            "dft_decomp",
        ], "decomp_method must be one of ['moving_avg', 'dft_decomp']."

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.top_k = top_k
        self.dropout = dropout
        self.channel_independence = channel_independence
        self.decomp_method = decomp_method
        self.moving_avg = moving_avg
        self.downsampling_layers = downsampling_layers
        self.downsampling_window = downsampling_window
        self.apply_nonstationary_norm = apply_nonstationary_norm

        # set up the model
        self.model = _TimeMixer(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            top_k=self.top_k,
            channel_independence=self.channel_independence,
            decomp_method=self.decomp_method,
            moving_avg=self.moving_avg,
            downsampling_layers=self.downsampling_layers,
            downsampling_window=self.downsampling_window,
            apply_nonstationary_norm=self.apply_nonstationary_norm,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )

        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """Prepares input batch for training."""
        indices, X, missing_mask, X_ori, indicating_mask = self._send_data_to_given_device(data)
        return {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        """Prepares input batch for validation."""
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """Prepares input batch for testing (inference)."""
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
        """Trains the model on the given dataset."""
        self.train_set = train_set

        # Step 1: Wrap training set
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Step 2: Wrap validation set if provided
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

        # Step 4: Load best model after training
        self.model.load_state_dict(self.best_model_dict)

        # Step 5: Save model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
