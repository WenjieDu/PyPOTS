"""
The implementation of FiLM for the partially-observed time-series forecasting task.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch

from ...nn.modules.loss import MSE, Criterion
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ..base import BaseNNForecaster
from .core import _FiLM


class FiLM(BaseNNForecaster):
    """The PyTorch implementation of the FiLM model :cite:`zhou2022film` for time series forecasting.

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

    window_size :
        A list including the window sizes for the HiPPO projection layers.

    multiscale :
        A list including the multiscale factors for the HiPPO projection layers.

    dropout :
        The dropout ratio for the HiPPO projection layers.
        It only works when mode_type == 1.

    mode_type :
        The mode type of the SpectralConv1d layers.
        It has to be one of [0, 1, 2].

    d_model :
        The dimension of the model.

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
        The device for the model to run on, as a string, a :class:`torch.device` object, or a list of them.
        If None, PyPOTS selects CUDA, then Ascend NPU, then CPU, depending on availability.
        Use ``"cuda:0"``, ``"npu:0"``, or ``"cpu"`` to select a specific backend/device.
        Ascend NPU requires a compatible torch-npu and CANN environment; model and operator support may vary.
        For compatible neural-network models, a CUDA device list uses DataParallel; for DDP training,
        pass a single device per process and use :func:`pypots.utils.distributed.run_ddp_fit`.

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
        window_size: list,
        multiscale: list,
        modes1: int = 32,
        dropout: float = 0.5,
        mode_type: int = 0,
        d_model: int = 128,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int | None = None,
        training_loss: Criterion | type = MSE,
        validation_metric: Criterion | type = MSE,
        optimizer: Optimizer | type = Adam,
        num_workers: int = 0,
        device: str | torch.device | list | None = None,
        saving_path: str | None = None,
        model_saving_strategy: str | None = "best",
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

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

        # model hyperparameters
        self.window_size = window_size
        self.multiscale = multiscale
        self.modes1 = modes1
        self.dropout = dropout
        self.mode_type = mode_type
        self.d_model = d_model

        # set up the model
        self.model = _FiLM(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_pred_steps=self.n_pred_steps,
            n_pred_features=self.n_pred_features,
            window_size=self.window_size,
            multiscale=self.multiscale,
            modes1=self.modes1,
            ratio=self.dropout,
            mode_type=self.mode_type,
            d_model=self.d_model,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())
