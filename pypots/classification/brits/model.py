"""
PyTorch BRITS model for both the time-series imputation task and the classification task.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.classification.base import BaseNNClassifier
from pypots.classification.brits.dataset import DatasetForBRITS
from pypots.imputation.brits.model import (
    RITS as imputation_RITS,
    _BRITS as imputation_BRITS,
)


class RITS(imputation_RITS):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        device: Union[str, torch.device],
    ):
        super().__init__(n_steps, n_features, rnn_hidden_size, device)
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(self.rnn_hidden_size, n_classes)

    def forward(self, inputs: dict, direction: str = "forward") -> dict:
        ret_dict = super().forward(inputs, direction)
        logits = self.classifier(ret_dict["final_hidden_state"])
        ret_dict["prediction"] = torch.softmax(logits, dim=1)
        return ret_dict


class _BRITS(imputation_BRITS, nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        classification_weight: float,
        reconstruction_weight: float,
        device: Union[str, torch.device],
    ):
        super().__init__(n_steps, n_features, rnn_hidden_size, device)
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes

        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, n_classes, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, n_classes, device)
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

    def impute(self, inputs: dict) -> torch.Tensor:
        return super().impute(inputs)

    def classify(self, inputs: dict) -> torch.Tensor:
        ret_f = self.rits_f(inputs, "forward")
        ret_b = self._reverse(self.rits_b(inputs, "backward"))
        classification_pred = (ret_f["prediction"] + ret_b["prediction"]) / 2
        return classification_pred

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
        ret_f = self.rits_f(inputs, "forward")
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        ret_f["classification_loss"] = F.nll_loss(
            torch.log(ret_f["prediction"]), inputs["label"]
        )
        ret_b["classification_loss"] = F.nll_loss(
            torch.log(ret_b["prediction"]), inputs["label"]
        )
        consistency_loss = self._get_consistency_loss(
            ret_f["imputed_data"], ret_b["imputed_data"]
        )
        classification_loss = (
            ret_f["classification_loss"] + ret_b["classification_loss"]
        ) / 2
        reconstruction_loss = (
            ret_f["reconstruction_loss"] + ret_b["reconstruction_loss"]
        ) / 2

        loss = (
            consistency_loss
            + reconstruction_loss * self.reconstruction_weight
            + classification_loss * self.classification_weight
        )

        results = {
            "consistency_loss": consistency_loss,
            "classification_loss": classification_loss,
            "reconstruction_loss": reconstruction_loss,
            "loss": loss,
        }
        return results


class BRITS(BaseNNClassifier):
    """BRITS implementation of BaseClassifier.

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
        n_classes: int,
        classification_weight: float = 1,
        reconstruction_weight: float = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
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
            learning_rate,
            weight_decay,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        self.model = _BRITS(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.n_classes,
            self.classification_weight,
            self.reconstruction_weight,
            self.device,
        )
        self.model = self.model.to(self.device)
        self._print_model_size()

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
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
            label,
        ) = map(lambda x: x.to(self.device), data)

        # assemble input data
        inputs = {
            "indices": indices,
            "label": label,
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
        # fetch data
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
        ) = map(lambda x: x.to(self.device), data)

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
                "deltas": back_deltas,
                "missing_mask": back_missing_mask,
            },
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

        val_set : dict or str or None,
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
        training_set = DatasetForBRITS(train_set, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            val_set = DatasetForBRITS(val_set, file_type=file_type)
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
        self.auto_save_model_if_necessary(training_finished=True)

    def classify(self, X: Union[dict, str], file_type: str = "h5py"):
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
        test_set = DatasetForBRITS(X, return_labels=False, file_type=file_type)
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
                classification_pred = self.model.classify(inputs)
                prediction_collector.append(classification_pred)

        predictions = torch.cat(prediction_collector)
        return predictions.cpu().detach().numpy()
