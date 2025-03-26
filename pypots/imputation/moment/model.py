"""
The implementation of MOMENT for the partially-observed time-series imputation task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from .core import _MOMENT
from ..base import BaseNNImputer
from ..saits.data import DatasetForSAITS
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class MOMENT(BaseNNImputer):
    """The PyTorch implementation of the MOMENT model :cite:`goswami2024moment`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.
    patch_size :
        The patch length for patch embedding.

    patch_stride :
        The stride for patch embedding.

    transformer_backbone :
        The backbone of the transformer model. It has to be one of ["t5-small","t5-base","t5-large","t5-3b","t5-11b",
        "google/flan-t5-small","google/flan-t5-base","google/flan-t5-large","google/flan-t5-xl","google/flan-t5-xxl"].

    transformer_type :
        The type of the transformer model. It has to be one of ["encoder_only","decoder_only","encoder_decoder"].

    n_layers :
        The number of layers in the transformer backbone.

    d_ffn :
        The hidden size of the feed-forward network.

    d_model :
        The hidden size of the model backbone.

    d_ffn :
        The hidden size of the feed-forward network .

    dropout :
        The dropout rate for the model.

    head_dropout :
        The dropout rate for the head of the model.

    finetuning_mode :
        The fine-tuning mode for the model. It has to be one of ["linear-probing","end-to-end","zero-shot"].

    revin_affine :
        Whether to use affine transformation in the RevIn module.

    add_positional_embedding :
        Whether to add positional embedding in the model.

    value_embedding_bias :
        Whether to add bias in the value embedding.

    orth_gain :
        The gain for the orthogonal initialization.

    mask_ratio :
        The ratio of the mask for the model.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss:
        The customized loss function designed by users for training the model.
        If not given, will use the default loss as claimed in the original paper.

    validation_metric:
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
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        patch_size: int,
        patch_stride: int,
        transformer_backbone: str,
        transformer_type: str,
        n_layers: int,
        d_ffn: int,
        d_model: int,
        dropout: float,
        head_dropout: float,
        finetuning_mode: str,
        revin_affine: bool,
        add_positional_embedding: bool,
        value_embedding_bias: bool,
        orth_gain: float,
        mask_ratio: float = 0.3,
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
            enable_amp=True,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hyperparameters
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.transformer_backbone = transformer_backbone
        self.transformer_type = transformer_type
        self.head_dropout = head_dropout
        self.finetuning_mode = finetuning_mode
        self.revin_affine = revin_affine
        self.add_positional_embedding = add_positional_embedding
        self.value_embedding_bias = value_embedding_bias
        self.orth_gain = orth_gain
        self.mask_ratio = mask_ratio
        # set up the model
        self.model = _MOMENT(
            n_steps=self.n_steps,
            n_features=self.n_features,
            transformer_backbone=self.transformer_backbone,
            transformer_type=self.transformer_type,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            finetuning_mode=self.finetuning_mode,
            revin_affine=self.revin_affine,
            add_positional_embedding=self.add_positional_embedding,
            value_embedding_bias=self.value_embedding_bias,
            orth_gain=self.orth_gain,
            mask_ratio=self.mask_ratio,
            device=self.device,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
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
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
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
        # Step 1: wrap the input data with classes Dataset and DataLoader
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

        # Step 2: train the model and freeze it
        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
