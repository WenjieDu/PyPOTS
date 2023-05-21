"""
The implementation of BRITS for the partially-observed time-series imputation task.

Refer to the paper "Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS. The bugs in the original implementation
are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Tuple, Union, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DatasetForBRITS
from .modules import TemporalDecay, FeatureRegression
from ..base import BaseNNImputer
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.metrics import cal_mae


class RITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    device :
        specify running the model on which device, CPU/GPU

    rnn_cell :
        the LSTM cell to model temporal data

    temp_decay_h :
        the temporal decay module to decay RNN hidden state

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist_reg :
        the temporal-regression module to project RNN hidden state into the raw feature space

    feat_reg :
        the feature-regression module

    combining_weight :
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    device :
        specify running the model on which device, CPU/GPU

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def impute(
        self, inputs: dict, direction: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The imputation function.
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data :
            [batch size, sequence length, feature number]

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        values = inputs[direction]["X"]  # feature values
        masks = inputs[direction]["missing_mask"]  # missing masks
        deltas = inputs[direction]["deltas"]  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(values.device)

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += cal_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += cal_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += cal_mae(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, hidden_states, reconstruction_loss

    def forward(self, inputs: dict, direction: str = "forward") -> dict:
        """Forward processing of the NN module.
        Parameters
        ----------
        inputs :
            The input data.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        dict,
            A dictionary includes all results.

        """
        imputed_data, hidden_state, reconstruction_loss = self.impute(inputs, direction)
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        ret_dict = {
            "consistency_loss": torch.tensor(
                0.0, device=imputed_data.device
            ),  # single direction, has no consistency loss
            "reconstruction_loss": reconstruction_loss,
            "imputed_data": imputed_data,
            "final_hidden_state": hidden_state,
        }
        return ret_dict


class _BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rits_f: RITS object
        the forward RITS model

    rits_b: RITS object
        the backward RITS model

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, device)

    @staticmethod
    def _get_consistency_loss(
        pred_f: torch.Tensor, pred_b: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f :
            The imputation from the forward RITS.

        pred_b :
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.

        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def _reverse(ret: dict) -> dict:
        """Reverse the array values on the time dimension in the given dictionary.

        Parameters
        ----------
        ret :

        Returns
        -------
        dict,
            A dictionary contains values reversed on the time dimension from the given dict.

        """

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def forward(self, inputs: dict, training: bool = True) -> dict:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs :
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        # Results from the forward RITS.
        ret_f = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2

        if not training:
            # if not in training mode, return the classification result only
            return {
                "imputed_data": imputed_data,
            }

        consistency_loss = self._get_consistency_loss(
            ret_f["imputed_data"], ret_b["imputed_data"]
        )

        # `loss` is always the item for backward propagating to update the model
        loss = (
            consistency_loss
            + ret_f["reconstruction_loss"]
            + ret_b["reconstruction_loss"]
        )

        results = {
            "imputed_data": imputed_data,
            "consistency_loss": consistency_loss,
            "loss": loss,  # will be used for backward propagating to update the model
        }

        return results


class BRITS(BaseNNImputer):
    """The PyTorch implementation of the BRITS model :cite:`cao2018BRITS`.

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
        self.model = _BRITS(
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
        training_set = DatasetForBRITS(
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
            val_set = DatasetForBRITS(val_set, return_labels=False, file_type=file_type)
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
        test_set = DatasetForBRITS(X, return_labels=False, file_type=file_type)
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
