"""
PyTorch SAITS model for the time-series imputation task.

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Tuple, Union, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.data.base import BaseDataset
from pypots.imputation.base import BaseNNImputer
from pypots.imputation.saits.data import DatasetForSAITS
from pypots.imputation.transformer.modules import EncoderLayer, PositionalEncoding
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer
from pypots.utils.metrics import cal_mae


class _SAITS(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_time: int,
        d_feature: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool = True,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                    diagonal_attention_mask,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                    diagonal_attention_mask,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def _process(self, inputs: dict) -> Tuple[torch.Tensor, list]:
        X, masks = inputs["X"], inputs["missing_mask"]
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely, term e in the math equation
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        # namely term eta
        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3

        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def impute(self, inputs: dict) -> torch.Tensor:
        imputed_data, _ = self._process(inputs)
        return imputed_data

    def forward(self, inputs: dict) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]
        ORT_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self._process(inputs)

        ORT_loss += cal_mae(X_tilde_1, X, masks)
        ORT_loss += cal_mae(X_tilde_2, X, masks)
        ORT_loss += cal_mae(X_tilde_3, X, masks)
        ORT_loss /= 3

        MIT_loss = cal_mae(X_tilde_3, inputs["X_intact"], inputs["indicating_mask"])

        # `loss` is always the item for backward propagating to update the model
        loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss

        results = {
            "imputed_data": imputed_data,
            "ORT_loss": ORT_loss,
            "MIT_loss": MIT_loss,
            "loss": loss,  # will be used for backward propagating to update the model
        }
        return results


class SAITS(BaseNNImputer):
    """The PyTorch implementation of the SAITS model :cite:`du2023SAITS`.

    Parameters
    ----------
    n_steps : int,
        The number of time steps in the time-series data sample.

    n_features : int,
        The number of features in the time-series data sample.

    n_layers : int,
        The number of layers in the 1st and 2nd DMSA blocks in the SAITS model.

    d_model : int,
        The dimension of the model's backbone.
        It is the input dimension of the multi-head DMSA layers.

    d_inner : int,
        The dimension of the layer in the Feed-Forward Networks (FFN).

    n_heads : int,
        The number of heads in the multi-head DMSA mechanism.
        ``d_model`` must be divisible by ``n_heads``, and the result should be equal to ``d_k``.

    d_k : int,
        The dimension of the `keys` (K) and the `queries` (Q) in the DMSA mechanism.
        ``d_k`` should be the result of ``d_model`` divided by ``n_heads``. Although ``d_k`` can be directly calculated
        with given ``d_model`` and ``n_heads``, we want it be explicitly given together with ``d_v`` by users to ensure
        users be aware of them and to avoid any potential mistakes.

    d_v : int,
        The dimension of the `values` (V) in the DMSA mechanism.

    dropout : float, 0<= ``dropout`` <1,
        The dropout rate for all fully-connected layers in the model.

    attn_dropout : float, 0<= ``attn_dropout`` <1,
        The dropout rate for DMSA.

    diagonal_attention_mask : bool, default = True,
        Whether to apply a diagonal attention mask to the self-attention mechanism.
        If so, the attention layers will use DMSA. Otherwise, the attention layers will use the original.

    ORT_weight : float, default = 1.0,
        The weight for the ORT loss.

    MIT_weight : float, default = 1.0,
        The weight for the MIT loss.

    batch_size : int, default = 32,
        The batch size for training and evaluating the model.

    epochs : int, default = 100,
        The number of epochs for training the model.

    patience : int, default = None,
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer : ``pypots.optim.base.Optimizer``, default = ``pypots.optim.Adam()``,
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers : int, default = 0,
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first (will use the GPU with device number 0 only by default),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, default = None,
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str or None, None or "best" or "better" , default = "best",
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    Attributes
    ----------
    model : object,
        The underlying SAITS model.

    optimizer : object,
        The optimizer for model training.

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
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
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
        self.model = self.model.to(self.device)
        self.print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble the given data into a dictionary for training input.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model training.
        """

        indices, X_intact, X, missing_mask, indicating_mask = map(
            lambda x: x.to(self.device), data
        )

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data) -> dict:
        """Assemble the given data into a dictionary for validating input.

        Notes
        -----
        The validating data assembling processing is the same as training data assembling.


        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model validating.
        """
        indices, X, missing_mask = map(lambda x: x.to(self.device), data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def _assemble_input_for_testing(self, data) -> dict:
        """Assemble the given data into a dictionary for testing input.

        Notes
        -----
        The testing data assembling processing is the same as training data assembling.

        Parameters
        ----------
        data : list,
            A list containing data fetched from Dataset by Dataloader.

        Returns
        -------
        inputs : dict,
            A python dictionary contains the input data for model testing.
        """
        return self._assemble_input_for_validating(data)

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "h5py",
    ) -> None:
        """Train the imputer on the given data.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        val_set : dict or str,
            The dataset for model validating, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for validating, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type : str, default = "h5py",
            The type of the given file if train_set and val_set are path strings.

        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForSAITS(train_set, file_type=file_type)
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

            val_set = BaseDataset(val_set, file_type=file_type)
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

    def impute(
        self,
        X: Union[dict, str],
        file_type="h5py",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py",
            The type of the given file if X is a path string.

        Returns
        -------
        imputed_data : array-like, shape [n_samples, sequence length (time steps), n_features],
            The imputed data.

        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(X, return_labels=False, file_type=file_type)
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
                imputed_data = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        # Step 3: output collection and return
        imputation_collector = torch.cat(imputation_collector)
        imputed_data = imputation_collector.cpu().detach().numpy()
        return imputed_data
