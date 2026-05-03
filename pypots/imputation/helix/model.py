"""
The implementation of HELIX for the partially-observed time-series imputation task.
"""

# Created by Fengming Zhang <milaogou@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from .core import _HELIX
from ..base import BaseNNImputer
from ..saits.data import DatasetForSAITS
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class HELIX(BaseNNImputer):
    """The PyTorch implementation of the HELIX: Hybrid Encoding with Learnable Identity
    and Cross-dimensional Synthesis for Time Series Imputation :cite:`zhang2026helix`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    d_pe :
        The dimension of the positional encoding for temporal dimension.
        Total embedding dimension will be pe_dim + feature_embed_dim + 2 (data + temporal_pe + feature_id + mask).

    d_feature_embed :
        The dimension of the learnable feature identity embedding.

    d_model :
        The dimension of the model's hidden states.

    n_heads :
        The number of attention heads.
        ``d_model`` must be divisible by ``n_heads``.

    n_layers :
        The number of hybrid encoding layers.

    dropout :
        The dropout rate for all layers.

    ORT_weight :
        The weight for the Observed Reconstruction Task (ORT) loss.

    MIT_weight :
        The weight for the Masked Imputation Task (MIT) loss.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss :
        The customized loss function designed by users for training the model.
        If not given, will use MAE as default.

    validation_metric :
        The customized metric function designed by users for validating the model.
        If not given, will use MSE as default.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process.

    device :
        The device for the model to run on.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].

    verbose :
        Whether to print out the training logs during the training process.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        d_pe: int = 16,
        d_feature_embed: int = 1,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        ORT_weight: float = 1.0,
        MIT_weight: float = 1.0,
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

        # Check d_model divisibility
        if d_model % n_heads != 0:
            logger.warning(f"‼️ d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            d_model = n_heads * (d_model // n_heads)
            logger.warning(f"⚠️ d_model is adjusted to {d_model}")

        self.n_steps = n_steps
        self.n_features = n_features
        self.d_pe = d_pe
        self.d_feature_embed = d_feature_embed
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # Set up the model
        self.model = _HELIX(
            n_steps=n_steps,
            n_features=n_features,
            d_pe=d_pe,
            d_feature_embed=d_feature_embed,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ORT_weight=ORT_weight,
            MIT_weight=MIT_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # Set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(lr=self.lr)
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble input data for training."""
        indices, X, missing_mask, X_ori, indicating_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        """Assemble input data for validation."""
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        """Assemble input data for testing."""
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the HELIX model.

        Parameters
        ----------
        train_set :
            The training dataset.

        val_set :
            The validation dataset.

        file_type :
            The type of the data file if train_set/val_set are file paths.
        """
        # Create datasets
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
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
            val_dataset = DatasetForSAITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Train the model with LR scheduling
        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Save the model
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The dataset for testing.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        result_dict :
            The dictionary containing the imputation results.
        """
        result_dict = super().predict(test_set, file_type)
        return result_dict
