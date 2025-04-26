"""
The implementation of Nonstationary-Transformer for the partially-observed time-series anomaly detection task.

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from ..base import BaseNNDetector
from ...data.checking import key_in_data_set
from ...imputation.nonstationary_transformer.core import _NonstationaryTransformer
from ...imputation.saits.data import DatasetForSAITS
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class NonstationaryTransformer(BaseNNDetector):
    """The PyTorch implementation of the Nonstationary-Transformer model for the anomaly detection task.
    Originally proposed by Liu et al. in :cite:`liu2022nonstationary`.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    anomaly_rate : float
        The estimated anomaly rate in the dataset, within (0, 1). Used for thresholding.

    n_layers : int
        The number of layers in the NonstationaryTransformer model.

    d_model : int
        The dimension of the model.

    n_heads : int
        The number of attention heads.

    d_ffn : int
        The dimension of the feed-forward network.

    d_projector_hidden : list
        Dimensions of hidden layers in MLP projectors.

    n_projector_hidden_layers : int
        Number of hidden layers in MLP projectors.

    dropout : float, optional
        Dropout rate for the model.

    attn_dropout : float, optional
        Dropout rate in the attention mechanism.

    ORT_weight : float, optional
        Weight for ORT loss.

    MIT_weight : float, optional
        Weight for MIT loss.

    batch_size : int, optional
        Batch size for training and evaluation.

    epochs : int, optional
        Total number of training epochs.

    patience : int, optional
        Early stopping patience. Disabled if None.

    training_loss : Criterion or type, optional
        Loss function for training. Defaults to MAE.

    validation_metric : Criterion or type, optional
        Metric for validation. Defaults to MSE.

    optimizer : Optimizer or type, optional
        Optimizer class or instance.

    num_workers : int, optional
        Number of subprocesses for data loading.

    device : str, torch.device, or list, optional
        Device(s) to run the model.

    saving_path : str, optional
        Path to save model checkpoints and logs.

    model_saving_strategy : str or None, optional
        Saving strategy: None, "best", "better", or "all".

    verbose : bool, optional
        Whether to print training logs.
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
        d_projector_hidden: list,
        n_projector_hidden_layers: int,
        dropout: float = 0,
        attn_dropout: float = 0,
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
        assert len(d_projector_hidden) == n_projector_hidden_layers, (
            f"The length of d_projector_hidden should be equal to n_projector_hidden_layers, "
            f"but got {len(d_projector_hidden)} and {n_projector_hidden_layers}."
        )

        # Store model configuration
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ffn = d_ffn
        self.d_projector_hidden = d_projector_hidden
        self.n_projector_hidden_layers = n_projector_hidden_layers
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # Instantiate the underlying NonstationaryTransformer model
        self.model = _NonstationaryTransformer(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ffn=self.d_ffn,
            d_projector_hidden=self.d_projector_hidden,
            n_projector_hidden_layers=self.n_projector_hidden_layers,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
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
        Trains the model on the given dataset.
        """
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

        # Step 4: Restore the best model from training
        self.model.load_state_dict(self.best_model_dict)

        # Step 5: Save model if needed
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
