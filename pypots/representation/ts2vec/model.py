"""
The implementation of TS2Vec for the partially-observed time-series representation task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _TS2Vec
from ..base import BaseNNRepresentor
from ...data.dataset.base import BaseDataset
from ...nn.modules.loss import Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer


class TS2Vec(BaseNNRepresentor):
    """The PyTorch implementation of the TS2Vec model :cite:`yue2022ts2vec`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_output_dims :
        The number of output dimensions for the representation of the time-series data sample.

    d_hidden :
        The number of hidden dimensions for the TS2VEC encoder.

    n_layers :
        The number of layers for the TS2VEC encoder.

    mask_mode :
        The mode for generating the mask for the TS2VEC encoder.
        It has to be one of ['binomial', 'continuous', 'all_true', 'all_false', 'mask_last'].

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

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
        n_output_dims: int,
        d_hidden: int,
        n_layers: int,
        mask_mode: str = "binomial",
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            training_loss=Criterion,
            validation_metric=Criterion,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_output_dims = n_output_dims
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.mask_mode = mask_mode

        # set up the model
        self.model = _TS2Vec(
            self.n_steps,
            self.n_features,
            self.n_output_dims,
            self.d_hidden,
            self.n_layers,
            self.mask_mode,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        indices, X, missing_mask = self._send_data_to_given_device(data)
        missing_mask = missing_mask.to(torch.bool)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": torch.masked_fill(X, ~missing_mask, torch.nan),
        }
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        train_dataset = BaseDataset(
            train_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = None
        if val_set is not None:
            val_dataset = BaseDataset(
                val_set,
                return_X_ori=False,
                return_X_pred=False,
                return_y=False,
                file_type=file_type,
            )
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

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        mask: str = None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
    ) -> dict:
        result_dict = super().predict(
            test_set,
            file_type=file_type,
            mask=mask,
            encoding_window=encoding_window,
            causal=causal,
            sliding_length=sliding_length,
            sliding_padding=sliding_padding,
        )
        return result_dict

    def represent(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        mask: str = None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
    ) -> np.ndarray:
        results = super().represent(
            test_set,
            file_type=file_type,
            mask=mask,
            encoding_window=encoding_window,
            causal=causal,
            sliding_length=sliding_length,
            sliding_padding=sliding_padding,
        )
        return results
