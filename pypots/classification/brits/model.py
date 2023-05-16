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
from pypots.classification.brits.data import DatasetForBRITS
from pypots.classification.brits.modules import RITS
from pypots.imputation.brits.model import (
    _BRITS as imputation_BRITS,
)
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer


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
    """The PyTorch implementation of the BRITS model :cite:`cao2018BRITS`.

    Parameters
    ----------
    n_steps : int,
        The number of time steps in the time-series data sample.

    n_features : int,
        The number of features in the time-series data sample.

    n_classes : int,
        The number of classes in the classification task.

    rnn_hidden_size : int,
        The size of the RNN hidden state.

    classification_weight : float, default = 1,
        The loss weight for the classification task.

    reconstruction_weight : float, default = 1,
        The loss weight for the reconstruction task.

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

    model_saving_strategy : str, "best" or "better" , default = "best",
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    Attributes
    ----------
    model : object,
        The underlying BRITS model.

    optimizer : object,
        The optimizer for model training.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_classes: int,
        rnn_hidden_size: int,
        classification_weight: float = 1,
        reconstruction_weight: float = 1,
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
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        # set up the model
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
        self._auto_save_model_if_necessary(training_finished=True)

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
