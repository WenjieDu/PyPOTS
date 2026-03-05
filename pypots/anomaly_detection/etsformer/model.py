"""
The implementation of ETSformer for the partially-observed time-series anomaly detection task.

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...imputation.etsformer.core import _ETSformer
from ...imputation.saits.data import DatasetForSAITS
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class ETSformer(BaseNNDetector):
    """The PyTorch implementation of the ETSformer model for the anomaly detection task.

    ETSformer is originally proposed by Woo et al. in :cite:`woo2023etsformer`.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    anomaly_rate : float
        The estimated anomaly rate in the dataset, within the range (0, 1). Used for thresholding.

    n_encoder_layers : int
        The number of layers in the ETSformer encoder.

    n_decoder_layers : int
        The number of layers in the ETSformer decoder.

    d_model : int
        The dimension of the model.

    n_heads : int
        The number of heads in each layer of ETSformer.

    d_ffn : int
        The dimension of the feed-forward network.

    top_k : int
        Top-K Fourier bases.

    dropout : float, optional
        The dropout rate for the model.

    ORT_weight : float, optional
        The weight for the ORT loss.

    MIT_weight : float, optional
        The weight for the MIT loss.

    batch_size : int, optional
        The number of samples per batch during training and evaluation.

    epochs : int, optional
        Total number of training epochs.

    patience : int, optional
        Number of epochs to wait for improvement before triggering early stopping.

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
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        top_k: int,
        dropout: float = 0,
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

        # Store model hyperparameters
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.top_k = top_k
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # Initialize the ETSformer model
        self.model = _ETSformer(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            top_k=self.top_k,
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
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
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
        Trains the ETSformer model on the given dataset.
        """
        self.train_set = train_set

        # Step 1: Wrap training data
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Step 2: Wrap validation data if available
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

        # Step 4: Load best model
        self.model.load_state_dict(self.best_model_dict)

        # Step 5: Save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
