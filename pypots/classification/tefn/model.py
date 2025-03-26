"""
The implementation of TEFN for the partially-observed time-series classification task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Union

import torch

from .core import _TEFN
from ..base import BaseNNClassifier
from ...nn.modules.loss import Criterion, CrossEntropy
from ...optim.adam import Adam
from ...optim.base import Optimizer


class TEFN(BaseNNClassifier):
    """The PyTorch implementation of the TEFN classification model :cite:`zhan2024tefn`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_classes :
        The number of classes in the classification task.

    n_fod :
        The number of FOD (frame of discernment) in the TEFN model.

    dropout :
        The dropout rate for the model.

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
        If not given, will use the default loss from the original paper as the metric.

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
        n_classes: int,
        n_fod: int,
        dropout: float = 0,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = CrossEntropy,
        validation_metric: Union[Criterion, type] = CrossEntropy,
        optimizer: Optimizer = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            n_classes=n_classes,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            training_loss=training_loss,
            validation_metric=validation_metric,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_fod = n_fod
        self.dropout = dropout

        # set up the model
        self.model = _TEFN(
            n_classes=self.n_classes,
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_fod=self.n_fod,
            dropout=self.dropout,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())
