"""
The implementation of Transformer for the partially-observed time-series anomaly detection task.
"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...imputation.transformer.core import _Transformer
from ...imputation.saits.data import DatasetForSAITS
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class Transformer(BaseNNDetector):
    """The PyTorch implementation of the Transformer model for the anomaly detection task.

    Transformer is originally proposed by Vaswani et al. in :cite:`vaswani2017Transformer`,
    and gets re-implemented for partially-observed time-series modeling by Du et al. in :cite:`du2023SAITS`.
    Here we adapt it specifically for anomaly detection tasks.

    Parameters
    ----------
    n_steps : int
        The number of time steps in each input time-series sample.

    n_features : int
        The number of features (dimensions) in each input time-series sample.

    anomaly_rate : float
        The expected anomaly rate within the dataset, between (0, 1). Used to determine detection thresholds.

    n_layers : int
        The number of stacked Transformer encoder layers.

    d_model : int
        The dimensionality of inputs and outputs inside the model's backbone.
        It is also the input dimension to the multi-head self-attention blocks.

    n_heads : int
        The number of parallel heads used in multi-head self-attention mechanisms.

    d_k : int
        The dimensionality of key and query vectors in the attention mechanism.
        Must satisfy d_model = n_heads * d_k.

    d_v : int
        The dimensionality of value vectors in the attention mechanism.

    d_ffn : int
        The dimensionality of the hidden layer inside the position-wise Feed-Forward Network (FFN).

    dropout : float, optional
        Dropout probability applied across fully connected layers. Default is 0.

    attn_dropout : float, optional
        Dropout probability applied inside attention mechanisms. Default is 0.

    ORT_weight : int, optional
        Weight coefficient for the ORT (Observation Reconstruction Task) loss component.

    MIT_weight : int, optional
        Weight coefficient for the MIT (Missingness Imputation Task) loss component.

    batch_size : int, optional
        Number of samples in each training batch. Default is 32.

    epochs : int, optional
        Maximum number of epochs to train the model. Default is 100.

    patience : int, optional
        Number of epochs to wait without improvement before early stopping is triggered.
        If None, early stopping is disabled.

    training_loss : Criterion or type, optional
        Loss function used for training. If not specified, defaults to Mean Absolute Error (MAE).

    validation_metric : Criterion or type, optional
        Metric used to evaluate model performance on validation set. Defaults to Mean Squared Error (MSE).

    optimizer : Optimizer or type, optional
        Optimizer used for training the model. Defaults to a custom implementation of Adam.

    num_workers : int, optional
        Number of worker subprocesses to use for data loading. 0 means no subprocesses (i.e., main process only).

    device : str, torch.device, or list, optional
        Device(s) on which the model runs, e.g., 'cuda:0', 'cpu', or list of CUDA devices for multi-GPU training.
        If None, the model automatically selects GPU if available, otherwise CPU.

    saving_path : str, optional
        Directory path for saving trained model checkpoints and TensorBoard logs. No saving if None.

    model_saving_strategy : str or None, optional
        Strategy for saving model checkpoints:
        - None: Do not save any model.
        - "best": Save only the best-performing model.
        - "better": Save model when validation performance improves.
        - "all": Save model at every epoch.

    verbose : bool, optional
        Whether to print detailed training logs during model training. Default is True.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0,
        attn_dropout: float = 0,
        ORT_weight: int = 1,
        MIT_weight: int = 1,
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
        """
        Initialize the Transformer anomaly detector.
        """
        # Initialize the parent class BaseNNDetector
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

        # Validate model structure: d_model must match n_heads * d_k
        if d_model != n_heads * d_k:
            logger.warning(
                f"‼️ d_model must equal n_heads * d_k. Received: d_model={d_model}, n_heads={n_heads}, d_k={d_k}."
            )
            d_model = n_heads * d_k
            logger.warning(f"⚠️ d_model is reset to {d_model}")

        # Save model configuration
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # Instantiate the Transformer model
        self.model = _Transformer(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            ORT_weight=self.ORT_weight,
            MIT_weight=self.MIT_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )

        # Move model to devices (CPU/GPU)
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
        Prepare input batch for training.

        Returns
        -------
        dict
            A dictionary with 'X', 'missing_mask', 'X_ori', and 'indicating_mask'.
        """
        indices, X, missing_mask, X_ori, indicating_mask = self._send_data_to_given_device(data)

        return {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        """
        Prepare input batch for validation.

        Returns
        -------
        dict
            Same as training input.
        """
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """
        Prepare input batch for testing.

        Returns
        -------
        dict
            A dictionary containing 'X' and 'missing_mask'.
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
        Train the Transformer model for anomaly detection.

        Parameters
        ----------
        train_set : dict or str
            Training dataset or path to it.

        val_set : dict or str, optional
            Validation dataset or path to it. Must include 'X_ori'.

        file_type : str, optional
            File type if loading from disk. Default is "hdf5".
        """
        self.train_set = train_set

        # Wrap training dataset
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Wrap validation dataset if available
        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for validation.")
            val_dataset = DatasetForSAITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Train the model and restore the best model
        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Save model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
