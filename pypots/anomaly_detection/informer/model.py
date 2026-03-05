"""
The implementation of Informer for the partially-observed time-series anomaly detection task.
"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...imputation.informer.core import _Informer
from ...imputation.saits.data import DatasetForSAITS
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class Informer(BaseNNDetector):
    """The PyTorch implementation of the Informer model :cite:zhou2021informer for the anomaly detection task.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    anomaly_rate : float
        The estimated anomaly rate in the dataset, within the range (0, 1). Used for thresholding.

    n_layers : int
        The number of layers in the Informer model.

    d_model : int
        The dimensionality of input and output feature vectors.

    n_heads : int
        The number of attention heads in each Transformer block.

    d_ffn : int
        The dimensionality of the feed-forward network hidden layer.

    factor : int
        The factor for controlling sparsity in the ProbSparse attention mechanism.

    dropout : float, optional
        Dropout rate applied throughout the model. Default is 0.

    ORT_weight : float, optional
        The weight for the ORT loss term during training. Default is 1.

    MIT_weight : float, optional
        The weight for the MIT loss term during training. Default is 1.

    batch_size : int, optional
        The number of samples per batch during training and evaluation. Default is 32.

    epochs : int, optional
        The total number of training epochs. Default is 100.

    patience : int, optional
        Number of epochs to wait for early stopping if no improvement. If None, early stopping is disabled.

    training_loss : Criterion or type, optional
        Loss function used during training. Defaults to MAE.

    validation_metric : Criterion or type, optional
        Metric function used for validation during training. Defaults to MSE.

    optimizer : Optimizer or type, optional
        The optimizer instance or optimizer class used for model training. Defaults to custom Adam optimizer.

    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 0.

    device : str, torch.device, or list, optional
        The device(s) on which the model will be trained or evaluated. Supports CPU, CUDA, and multi-GPU.

    saving_path : str, optional
        The path where model checkpoints and training logs are saved. No saving if None.

    model_saving_strategy : str or None, optional
        Strategy to save models during training: one of {None, "best", "better", "all"}.

    verbose : bool, optional
        Whether to print detailed logs during the training process. Default is True.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        factor: int,
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
        # Call base detector initialization
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

        # Save model hyperparameters
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.factor = factor
        self.dropout = dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # Instantiate the Informer model
        self.model = _Informer(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ffn=self.d_ffn,
            factor=self.factor,
            dropout=self.dropout,
            distil=False,  # distilling disabled
            ORT_weight=self.ORT_weight,
            MIT_weight=self.MIT_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )

        # Move model to target device (GPU/CPU)
        self._send_model_to_given_device()

        # Print model size
        self._print_model_size()

        # Initialize optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)

        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """
        Assemble input batch dictionary for model training.

        Parameters
        ----------
        data : list
            A list of batched data from the DataLoader.

        Returns
        -------
        dict
            Formatted input dictionary with keys 'X', 'missing_mask', 'X_ori', and 'indicating_mask'.
        """
        # Move data tensors to the correct device
        indices, X, missing_mask, X_ori, indicating_mask = self._send_data_to_given_device(data)

        return {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        """
        Assemble input batch dictionary for model validation.

        Parameters
        ----------
        data : list
            A list of batched validation data.

        Returns
        -------
        dict
            Same format as training input.
        """
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """
        Assemble input batch dictionary for model testing.

        Parameters
        ----------
        data : list
            A list of batched testing data.

        Returns
        -------
        dict
            Input dictionary containing 'X' and 'missing_mask' only.
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
        Train the Informer model on the provided datasets.

        Parameters
        ----------
        train_set : dict or str
            The training dataset or file path.

        val_set : dict or str, optional
            The validation dataset or file path. Must contain 'X_ori' if provided.

        file_type : str, optional
            File type if data is loaded from disk. Default is "hdf5".
        """
        # Save reference to training set
        self.train_set = train_set

        # Prepare training dataset and DataLoader
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle during training
            num_workers=self.num_workers,
        )

        # Prepare validation dataset and DataLoader if provided
        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for validation.")
            val_dataset = DatasetForSAITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No shuffle during validation
                num_workers=self.num_workers,
            )

        # Train the model
        self._train_model(train_dataloader, val_dataloader)

        # Load best model state
        self.model.load_state_dict(self.best_model_dict)

        # Save best model if configured
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
