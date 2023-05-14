"""
PyTorch GRU-D model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.classification.base import BaseNNClassifier
from pypots.classification.grud.data import DatasetForGRUD
from pypots.imputation.brits.modules import TemporalDecay
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer


class _GRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.device = device

        # create models
        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size
        )
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.classifier = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def classify(self, inputs: dict) -> torch.Tensor:
        values = inputs["X"]
        masks = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        hidden_state = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            inputs = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(inputs, hidden_state)

        logits = self.classifier(hidden_state)
        prediction = torch.softmax(logits, dim=1)
        return prediction

    def forward(self, inputs: dict) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        prediction = self.classify(inputs)
        classification_loss = F.nll_loss(torch.log(prediction), inputs["label"])
        results = {"prediction": prediction, "loss": classification_loss}
        return results


class GRUD(BaseNNClassifier):
    """GRU-D implementation of BaseClassifier.

    Attributes
    ----------
    model : object,
        The underlying GRU-D model.
    optimizer : object,
        The optimizer for model training.
    data_loader : object,
        The data loader for dataset loading.

    Parameters
    ----------
    rnn_hidden_size : int,
        The size of the RNN hidden state.
    optimizer : ``pypots.optim.base.Optimizer``, default = ``pypots.optim.Adam``(),
        The optimizer for model training.
        If not given, will use a default Adam optimizer.
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
        n_classes: int,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            n_classes,
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
        self.model = _GRUD(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.n_classes,
            self.device,
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: dict) -> dict:
        """Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        # fetch data
        indices, X, X_filledLOCF, missing_mask, deltas, empirical_mean, label = map(
            lambda x: x.to(self.device), data
        )

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "X_filledLOCF": X_filledLOCF,
            "missing_mask": missing_mask,
            "deltas": deltas,
            "empirical_mean": empirical_mean,
            "label": label,
        }
        return inputs

    def _assemble_input_for_validating(self, data: dict) -> dict:
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

    def _assemble_input_for_testing(self, data: dict) -> dict:
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
        indices, X, X_filledLOCF, missing_mask, deltas, empirical_mean = map(
            lambda x: x.to(self.device), data
        )

        inputs = {
            "indices": indices,
            "X": X,
            "X_filledLOCF": X_filledLOCF,
            "missing_mask": missing_mask,
            "deltas": deltas,
            "empirical_mean": empirical_mean,
        }

        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "h5py",
    ) -> None:
        """Train the classifier on the given data.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        val_set : dict or str,
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type : str, default = "h5py"
            The type of the given file if train_set and val_set are path strings.

        Returns
        -------
        self : object,
            Trained classifier.
        """
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGRUD(train_set, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            val_set = DatasetForGRUD(val_set, file_type=file_type)
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

    def classify(self, X: Union[dict, str], file_type: str = "h5py") -> np.ndarray:
        """Classify the input data with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py",
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Classification results of the given samples.
        """
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGRUD(X, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        prediction_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                prediction = self.model.classify(inputs)
                prediction_collector.append(prediction)

        predictions = torch.cat(prediction_collector)
        return predictions.cpu().detach().numpy()
