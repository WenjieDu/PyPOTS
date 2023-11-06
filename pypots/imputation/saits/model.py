"""
The implementation of SAITS for the partially-observed time-series imputation task.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DatasetForSAITS
from .modules import _SAITS
from ..base import BaseNNImputer
from ...data.base import BaseDataset
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from ...utils.metrics import cal_mae


class SAITS(BaseNNImputer):
    """The PyTorch implementation of the SAITS model :cite:`du2023SAITS`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_layers :
        The number of layers in the 1st and 2nd DMSA blocks in the SAITS model.

    d_model :
        The dimension of the model's backbone.
        It is the input dimension of the multi-head DMSA layers.

    d_inner :
        The dimension of the layer in the Feed-Forward Networks (FFN).

    n_heads :
        The number of heads in the multi-head DMSA mechanism.
        ``d_model`` must be divisible by ``n_heads``, and the result should be equal to ``d_k``.

    d_k :
        The dimension of the `keys` (K) and the `queries` (Q) in the DMSA mechanism.
        ``d_k`` should be the result of ``d_model`` divided by ``n_heads``. Although ``d_k`` can be directly calculated
        with given ``d_model`` and ``n_heads``, we want it be explicitly given together with ``d_v`` by users to ensure
        users be aware of them and to avoid any potential mistakes.

    d_v :
        The dimension of the `values` (V) in the DMSA mechanism.

    dropout :
        The dropout rate for all fully-connected layers in the model.

    attn_dropout :
        The dropout rate for DMSA.

    diagonal_attention_mask :
        Whether to apply a diagonal attention mask to the self-attention mechanism.
        If so, the attention layers will use DMSA. Otherwise, the attention layers will use the original.

    ORT_weight :
        The weight for the ORT loss.

    MIT_weight :
        The weight for the MIT loss.

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
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    References
    ----------
    .. [1] `Du, Wenjie, David Côté, and Yan Liu.
        "Saits: Self-attention-based imputation for time series".
        Expert Systems with Applications 219 (2023): 119619.
        <https://arxiv.org/pdf/2202.08516>`_

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0,
        attn_dropout: float = 0,
        diagonal_attention_mask: bool = True,
        ORT_weight: int = 1,
        MIT_weight: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        customized_loss_func: Callable = cal_mae,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # set up the model
        self.model = _SAITS(
            self.n_layers,
            self.n_steps,
            self.n_features,
            self.d_model,
            self.d_inner,
            self.n_heads,
            self.d_k,
            self.d_v,
            self.dropout,
            self.attn_dropout,
            self.diagonal_attention_mask,
            self.ORT_weight,
            self.MIT_weight,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the loss function
        self.customized_loss_func = customized_loss_func

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X_intact,
            X,
            missing_mask,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def _assemble_input_for_testing(self, data) -> dict:
        return self._assemble_input_for_validating(data)

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "h5py",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForSAITS(
            train_set, return_labels=False, file_type=file_type
        )
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if isinstance(val_set, str):
                with h5py.File(val_set, "r") as hf:
                    # Here we read the whole validation set from the file to mask a portion for validation.
                    # In PyPOTS, using a file usually because the data is too big. However, the validation set is
                    # generally shouldn't be too large. For example, we have 1 billion samples for model training.
                    # We won't take 20% of them as the validation set because we want as much as possible data for the
                    # training stage to enhance the model's generalization ability. Therefore, 100,000 representative
                    # samples will be enough to validate the model.
                    val_set = {
                        "X": hf["X"][:],
                        "X_intact": hf["X_intact"][:],
                        "indicating_mask": hf["indicating_mask"][:],
                    }

            val_set = BaseDataset(val_set, return_labels=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(training_finished=True)

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "h5py",
        diagonal_attention_mask: bool = True,
    ) -> dict:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(test_set, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        # Step 2: process the data with the model
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(
                    inputs, diagonal_attention_mask, training=False
                )
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)

        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        X: Union[dict, str],
        file_type="h5py",
    ) -> np.ndarray:
        logger.warning(
            "🚨DeprecationWarning: The method impute is deprecated. Please use `predict` instead."
        )
        results_dict = self.predict(X, file_type=file_type)
        return results_dict["imputation"]
