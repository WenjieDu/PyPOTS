"""
PyTorch MRNN model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Union, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DatasetForMRNN
from .module import FCN_Regression
from ..base import BaseNNImputer
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.metrics import cal_rmse


class _MRNN(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, device):
        super().__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.f_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.concated_hidden_project = nn.Linear(
            self.rnn_hidden_size * 2, self.feature_num
        )
        self.fcn_regression = FCN_Regression(feature_num, rnn_hidden_size)

    def gene_hidden_states(self, inputs, direction):
        X = inputs[direction]["X"]
        masks = inputs[direction]["missing_mask"]
        deltas = inputs[direction]["deltas"]
        device = X.device

        hidden_states_collector = []
        hidden_state = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)

        for t in range(self.seq_len):
            x = X[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            inputs = torch.cat([x, m, d], dim=1)
            if direction == "forward":
                hidden_state = self.f_rnn(inputs, hidden_state)
            else:
                hidden_state = self.b_rnn(inputs, hidden_state)
            hidden_states_collector.append(hidden_state)
        return hidden_states_collector

    def forward(self, inputs, training=True):
        hidden_states_f = self.gene_hidden_states(inputs, "forward")
        hidden_states_b = self.gene_hidden_states(inputs, "backward")[::-1]

        X = inputs["forward"]["X"]
        masks = inputs["forward"]["missing_mask"]

        reconstruction_loss = 0
        estimations = []
        for i in range(
            self.seq_len
        ):  # calculating estimation loss for times can obtain better results than once
            x = X[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(
                x, m, RNN_imputed_data
            )  # FCN estimation is output estimation
            reconstruction_loss += cal_rmse(FCN_estimation, x, m) + cal_rmse(
                RNN_estimation, x, m
            )
            estimations.append(FCN_estimation.unsqueeze(dim=1))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * X + (1 - masks) * estimations

        if not training:
            # if not in training mode, return the classification result only
            return {
                "imputed_data": imputed_data,
            }

        reconstruction_loss /= self.seq_len

        ret_dict = {
            "loss": reconstruction_loss,
            "imputed_data": imputed_data,
        }
        return ret_dict


class MRNN(BaseNNImputer):
    """The PyTorch implementation of the MRNN model :cite:`yoon2019MRNN`.

    Parameters
    ----------
    rnn_hidden_size :
        The size of the RNN hidden state, also the number of hidden units in the RNN cell.

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

    Attributes
    ----------
    model : :class:`torch.nn.Module`
        The underlying BRITS model.

    optimizer : :class:`pypots.optim.Optimizer`
        The optimizer for model training.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
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
        self.rnn_hidden_size = rnn_hidden_size

        # set up the model
        self.model = _MRNN(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.device,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "h5py",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForMRNN(
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
            val_set = DatasetForMRNN(val_set, return_labels=False, file_type=file_type)
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
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForMRNN(X, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
