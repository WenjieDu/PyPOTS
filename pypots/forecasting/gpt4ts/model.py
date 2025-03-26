"""
The implementation of GPT4TS for the partially-observed time-series forecasting task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from copy import deepcopy
from typing import Union, Optional

import torch

from .core import _GPT4TS
from ..base import BaseNNForecaster
from ...nn.modules.loss import Criterion, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class GPT4TS(BaseNNForecaster):
    """The PyTorch implementation of the GPT4TS forecasting model :cite:`zhou2023gpt4ts`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_pred_steps :
        The number of steps in the forecasting time series.

    n_pred_features :
        The number of features in the forecasting time series.

    term :
        The forecasting term, which can be either 'long' or 'short'.

    patch_size :
        The size of the patch for the patching mechanism.

    patch_stride :
        The stride for the patching mechanism.

    n_layers :
        The number of hidden layers to use in GPT2.

    train_gpt_mlp :
        Whether to train the MLP in GPT2 during tuning.

    d_ffn :
        The hidden size of the feed-forward network .

    dropout :
        The dropout rate for the model.

    embed :
        The embedding method for the model.

    freq :
        The frequency of the time-series data.
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
        n_pred_steps: int,
        n_pred_features: int,
        term: str,
        patch_size: int,
        patch_stride: int,
        n_layers: int,
        train_gpt_mlp: bool,
        d_ffn: int,
        dropout: float,
        embed: str = "fixed",
        freq="h",
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MSE,
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
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.term = term
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.train_gpt_mlp = train_gpt_mlp
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.embed = embed
        self.freq = freq

        # set up the model
        self.model = _GPT4TS(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_pred_steps=self.n_pred_steps,
            n_pred_features=self.n_pred_features,
            term=self.term,
            n_layers=self.n_layers,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            train_gpt_mlp=self.train_gpt_mlp,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            embed=self.embed,
            freq=self.freq,
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

    def _organize_content_to_save(self):
        from ...version import __version__ as pypots_version

        if isinstance(self.device, list):
            # to save a DataParallel model generically, save the model.module.state_dict()
            model_state_dict = deepcopy(self.model.module.state_dict())
        else:
            model_state_dict = deepcopy(self.model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items() if "gpt2" not in k}

        all_attrs = dict({})
        all_attrs["model_state_dict"] = model_state_dict
        all_attrs["pypots_version"] = pypots_version
        return all_attrs
