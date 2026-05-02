"""
The implementation of HELIX for the partially-observed time-series imputation task.
"""

# Created by MiBah Cat <milaogou@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _HELIX
from .data import DatasetForHELIX
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class HELIX(BaseNNImputer):
    """The PyTorch implementation of the HELIX model.
    
    HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis
    for Time Series Imputation.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    pe_dim :
        The dimension of the rotary positional encoding for temporal dimension.
        Total embedding dimension will be pe_dim + feature_embed_dim + 2 (data + temporal_pe + feature_id + mask).

    feature_embed_dim :
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

    lr :
        The learning rate for the optimizer.

    lr_decay_patience :
        The patience for learning rate decay. If validation loss doesn't improve for this many epochs,
        the learning rate will be halved.

    min_lr :
        The minimum learning rate. Learning rate will not decay below this value.

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
        pe_dim: int = 16,
        feature_embed_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        ORT_weight: float = 1.0,
        MIT_weight: float = 1.0,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        lr: float = 0.001,
        lr_decay_patience: Optional[int] = None,  # 默认改为None表示不启用
        min_lr: float = 1e-6,
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
            logger.warning(
                f"‼️ d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
            d_model = n_heads * (d_model // n_heads)
            logger.warning(f"⚠️ d_model is adjusted to {d_model}")

        self.n_steps = n_steps
        self.n_features = n_features
        self.pe_dim = pe_dim
        self.feature_embed_dim = feature_embed_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.lr = lr
        self.lr_decay_patience = lr_decay_patience
        self.min_lr = min_lr

        # Print model configuration
        self._print_model_configuration()

        # Set up the model
        self.model = _HELIX(
            n_steps=n_steps,
            n_features=n_features,
            pe_dim=pe_dim,
            feature_embed_dim=feature_embed_dim,
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

        # Set up learning rate scheduler (only if lr_decay_patience is enabled)
        self.lr_scheduler = None
        if self.lr_decay_patience is not None and self.lr_decay_patience > 0:
            # Access the internal PyTorch optimizer from PyPOTS wrapper
            torch_optimizer = self.optimizer.__dict__.get('torch_optimizer') or \
                             self.optimizer.__dict__.get('opt') or \
                             list(self.optimizer.__dict__.values())[0]
            
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                torch_optimizer,
                mode='min',
                factor=0.5,
                patience=self.lr_decay_patience,
                min_lr=self.min_lr,
                verbose=self.verbose
            )
            
            if self.verbose:
                logger.info(f"Learning rate scheduler enabled with patience={self.lr_decay_patience}")
        else:
            if self.verbose:
                logger.info("Learning rate scheduler disabled (lr_decay_patience not set)")

    def _print_model_configuration(self):
        """Print all model configuration parameters."""
        if self.verbose:
            logger.info("=" * 60)
            logger.info("HELIX Model Configuration:")
            logger.info("=" * 60)
            logger.info(f"Data dimensions:")
            logger.info(f"  - n_steps: {self.n_steps}")
            logger.info(f"  - n_features: {self.n_features}")
            logger.info(f"Model architecture:")
            logger.info(f"  - pe_dim: {self.pe_dim}")
            logger.info(f"  - feature_embed_dim: {self.feature_embed_dim}")
            logger.info(f"  - d_model: {self.d_model}")
            logger.info(f"  - n_heads: {self.n_heads}")
            logger.info(f"  - n_layers: {self.n_layers}")
            logger.info(f"  - dropout: {self.dropout}")
            logger.info(f"Training configuration:")
            logger.info(f"  - ORT_weight: {self.ORT_weight}")
            logger.info(f"  - MIT_weight: {self.MIT_weight}")
            logger.info(f"  - batch_size: {self.batch_size}")
            logger.info(f"  - epochs: {self.epochs}")
            logger.info(f"  - patience: {self.patience}")
            logger.info(f"Optimizer configuration:")
            logger.info(f"  - initial_lr: {self.lr}")
            if self.lr_decay_patience is not None and self.lr_decay_patience > 0:
                logger.info(f"  - lr_decay_patience: {self.lr_decay_patience}")
                logger.info(f"  - min_lr: {self.min_lr}")
            else:
                logger.info(f"  - lr_decay: disabled")
            logger.info(f"Other settings:")
            logger.info(f"  - num_workers: {self.num_workers}")
            logger.info(f"  - device: {self.device}")
            logger.info(f"  - model_saving_strategy: {self.model_saving_strategy}")
            logger.info("=" * 60)

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
        train_dataset = DatasetForHELIX(
            train_set, return_X_ori=False, return_y=False, file_type=file_type
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
            val_dataset = DatasetForHELIX(
                val_set, return_X_ori=True, return_y=False, file_type=file_type
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Train the model with LR scheduling
        self._train_model_with_lr_scheduling(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Save the model
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def _train_model_with_lr_scheduling(self, train_loader, val_loader=None):
        """Train model with learning rate scheduling."""
        self.optimizer.zero_grad()
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_loss = 0
            
            for idx, data in enumerate(train_loader):
                inputs = self._assemble_input_for_training(data)
                results = self.model.forward(inputs, calc_criterion=True)
                loss = results["loss"]
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_train_loss += loss.item()
            
            mean_train_loss = epoch_train_loss / len(train_loader)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = 0
                
                with torch.no_grad():
                    for idx, data in enumerate(val_loader):
                        inputs = self._assemble_input_for_validating(data)
                        results = self.model.forward(inputs, calc_criterion=True)
                        epoch_val_loss += results["metric"].item()
                
                mean_val_loss = epoch_val_loss / len(val_loader)
                
                # Step the learning rate scheduler (if enabled)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(mean_val_loss)
                
                # Get current learning rate
                torch_optimizer = self.optimizer.__dict__.get('torch_optimizer') or \
                                 self.optimizer.__dict__.get('opt') or \
                                 list(self.optimizer.__dict__.values())[0]
                current_lr = torch_optimizer.param_groups[0]['lr']
                
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"train_loss: {mean_train_loss:.4f}, "
                        f"val_loss: {mean_val_loss:.4f}, "
                        f"lr: {current_lr:.6f}"
                    )
                
                # Early stopping and model saving
                if mean_val_loss < self.best_loss:
                    self.best_loss = mean_val_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience_count = 0
                    
                    if self.model_saving_strategy == "better":
                        self._auto_save_model_if_necessary(confirm_saving=True)
                else:
                    self.patience_count += 1
                
                if self.patience is not None and self.patience_count >= self.patience:
                    if self.verbose:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"No improvement for {self.patience} epochs."
                        )
                    break
            else:
                # No validation set
                torch_optimizer = self.optimizer.__dict__.get('torch_optimizer') or \
                                 self.optimizer.__dict__.get('opt') or \
                                 list(self.optimizer.__dict__.values())[0]
                current_lr = torch_optimizer.param_groups[0]['lr']
                
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"train_loss: {mean_train_loss:.4f}, "
                        f"lr: {current_lr:.6f}"
                    )
                
                if self.model_saving_strategy == "all":
                    self._auto_save_model_if_necessary(confirm_saving=True)

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

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        imputation :
            Imputation results.
        """
        results = super().impute(test_set, file_type)
        return results