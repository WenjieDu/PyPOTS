"""
The implementation of PatchTST for the partially-observed time-series classification task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Union

import torch

from .core import _PatchTST
from ..base import BaseNNClassifier
from ...nn.modules.loss import Criterion, CrossEntropy
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class PatchTST(BaseNNClassifier):
    """The PyTorch implementation of the PatchTST classification model :cite:`nie2023patchtst`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_classes :
        The number of classes in the classification task.

    patch_size :
        The patch length for patch embedding.

    patch_stride :
        The stride for patch embedding.

    n_layers :
        The number of layers in the PatchTST model.

    d_model :
        The dimension of the model.

    n_heads :
        The number of heads in each layer of PatchTST.

    d_k :
        The dimension of the `keys` (K) and the `queries` (Q) in the DMSA mechanism.
        ``d_k`` should be the result of ``d_model`` divided by ``n_heads``. Although ``d_k`` can be directly calculated
        with given ``d_model`` and ``n_heads``, we want it be explicitly given together with ``d_v`` by users to ensure
        users be aware of them and to avoid any potential mistakes.

    d_v :
        The dimension of the `values` (V) in the DMSA mechanism.

    d_ffn :
        The dimension of the feed-forward network.

    dropout :
        The dropout rate for the model.

    attn_dropout :
        The dropout rate for the attention mechanism in the model.

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
        patch_size: int,
        patch_stride: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
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
        if d_model != n_heads * d_k:
            logger.warning(
                "‼️ d_model must = n_heads * d_k, it should be divisible by n_heads "
                f"and the result should be equal to d_k, but got d_model={d_model}, n_heads={n_heads}, d_k={d_k}"
            )
            d_model = n_heads * d_k
            logger.warning(f"⚠️ d_model is reset to {d_model} = n_heads ({n_heads}) * d_k ({d_k})")

        self.n_steps = n_steps
        self.n_features = n_features
        # model hyperparameters
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        # set up the model
        self.model = _PatchTST(
            n_classes=self.n_classes,
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ffn=self.d_ffn,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())
