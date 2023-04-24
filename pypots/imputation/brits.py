"""
PyTorch BRITS model for the time-series imputation task.

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import math
from typing import Tuple, Union, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from pypots.data.dataset_for_brits import DatasetForBRITS
from pypots.imputation.base import BaseNNImputer
from pypots.utils.metrics import cal_mae


class FeatureRegression(nn.Module):
    """The module used to capture the correlation between features for imputation.

    Attributes
    ----------
    W : tensor
        The weights (parameters) of the module.

    b : tensor
        The bias of the module.

    m (buffer) : tensor
        The mask matrix, a squire matrix with diagonal entries all zeroes while left parts all ones.
        It is applied to the weight matrix to mask out the estimation contributions from features themselves.
        It is used to help enhance the imputation performance of the network.

    Parameters
    ----------
    input_size : the feature dimension of the input
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        x : tensor,
            the input for processing

        Returns
        -------
        output: tensor,
            the processed result containing imputation from feature regression

        """
        output = F.linear(x, self.W * Variable(self.m), self.b)
        return output


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the original paper.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing
    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        delta : tensor, shape [batch size, sequence length, feature number]
            The time gaps.

        Returns
        -------
        gamma : array-like, same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps : int,
        sequence length (number of time steps)

    n_features : int,
        number of features (input dimensions)

    rnn_hidden_size : int,
        the hidden size of the RNN cell

    device : str, default=None,
        specify running the model on which device, CPU/GPU

    rnn_cell : torch.nn.module object
        the LSTM cell to model temporal data

    temp_decay_h : torch.nn.module object
        the temporal decay module to decay RNN hidden state

    temp_decay_x : torch.nn.module object
        the temporal decay module to decay data in the raw feature space

    hist_reg : torch.nn.module object
        the temporal-regression module to project RNN hidden state into the raw feature space

    feat_reg : torch.nn.module object
        the feature-regression module

    combining_weight : torch.nn.module object
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps : int,
        sequence length (number of time steps)

    n_features : int,
        number of features (input dimensions)

    rnn_hidden_size : int,
        the hidden size of the RNN cell

    device : str,
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
        inputs : dict,
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction : str, 'forward'/'backward'
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data : tensor,
            [batch size, sequence length, feature number]

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss : float tensor,
            reconstruction loss

        """
        values = inputs[direction]["X"]  # feature values
        masks = inputs[direction]["missing_mask"]  # missing masks
        deltas = inputs[direction]["deltas"]  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(self.device)

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
        inputs : dict,
            The input data.

        direction : string, 'forward'/'backward'
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
                0.0, device=self.device
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
    n_steps : int,
        sequence length (number of time steps)

    n_features : int,
        number of features (input dimensions)

    rnn_hidden_size : int,
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
        pred_f : tensor,
            The imputation from the forward RITS.

        pred_b : tensor,
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
        ret : dict

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

    def impute(self, inputs: dict) -> torch.Tensor:
        """Impute the missing data. Only impute, this is for test stage.

        Parameters
        ----------
        inputs : dict,
            A dictionary includes all input data.

        Returns
        -------
        array-like, the same shape with the input feature vectors.
            The feature vectors with missing part imputed.

        """
        imputed_data_f, _, _ = self.rits_f.impute(inputs, "forward")
        imputed_data_b, _, _ = self.rits_b.impute(inputs, "backward")
        imputed_data_b = {"imputed_data_b": imputed_data_b}
        imputed_data_b = self._reverse(imputed_data_b)["imputed_data_b"]
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data

    def forward(self, inputs: dict) -> dict:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        # Results from the forward RITS.
        ret_f = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        consistency_loss = self._get_consistency_loss(
            ret_f["imputed_data"], ret_b["imputed_data"]
        )
        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2

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
    """BRITS implementation

    Attributes
    ----------
    model : object,
        The underlying BRITS model.

    optimizer : object,
        The optimizer for model training.

    Parameters
    ----------
    rnn_hidden_size : int,
        The size of the RNN hidden state.

    learning_rate : float (0,1),
        The learning rate parameter for the optimizer.

    weight_decay : float in (0,1),
        The weight decay parameter for the optimizer.

    epochs : int,
        The number of training epochs.

    patience : int,
        The number of epochs with loss non-decreasing before early stopping the training.

    batch_size : int,
        The batch size of the training input.

    device :
        Run the model on which device.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        tb_file_saving_path: str = None,
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            learning_rate,
            weight_decay,
            num_workers,
            device,
            tb_file_saving_path,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.model = _BRITS(
            self.n_steps, self.n_features, self.rnn_hidden_size, self.device
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

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

        # fetch data
        indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = map(
            lambda x: x.to(self.device), data
        )

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
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
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
        training_set = DatasetForBRITS(train_set, file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        if val_set is None:
            self._train_model(training_loader)
        else:
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

            val_set = DatasetForBRITS(val_set)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            self._train_model(training_loader, val_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

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
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForBRITS(X)
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
                imputed_data = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
