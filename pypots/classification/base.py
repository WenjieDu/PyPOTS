"""
The base classes for PyPOTS classification models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..base import BaseModel, BaseNNModel
from ..nn.functional import autocast
from ..nn.modules.loss import Criterion, CrossEntropy
from ..nn.modules.metric import PR_AUC
from ..utils.logging import logger

try:
    import nni
except ImportError:
    pass


class BaseClassifier(BaseModel):
    """The abstract class for all PyPOTS classification models.

    Parameters
    ----------
    n_classes :
        The number of classes in the classification task.

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
        n_classes: int,
        device: Optional[Union[str, torch.device, list]] = None,
        enable_amp: bool = False,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            device=device,
            enable_amp=enable_amp,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        self.n_classes = n_classes

    @abstractmethod
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the classifier on the given data.

        Parameters
        ----------
        train_set :
            The dataset for model training, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for training, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        val_set :
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if train_set and val_set are path strings.

        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def classify(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Classify the input data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Classification results of the given samples.
        """

        raise NotImplementedError


class BaseNNClassifier(BaseNNModel):
    """The abstract class for all neural-network classification models in PyPOTS.

    Parameters
    ----------
    n_classes :
        The number of classes in the classification task.

    batch_size :
        Size of the batch input into the model for one step.

    epochs :
        Training epochs, i.e. the maximum rounds of the model to be trained with.

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

    enable_amp :
        Whether to enable automatic mixed precision (AMP), default as False.
        If the implemented model is based on LLMs that need large-scale operation and AMP, please set it as True.

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

    Notes
    -----
    Optimizers are necessary for training deep-learning neural networks, but we don't put  a parameter ``optimizer``
    here because some models (e.g. GANs) need more than one optimizer (e.g. one for generator, one for discriminator),
    and ``optimizer`` is ambiguous for them. Therefore, we leave optimizers as parameters for concrete model
    implementations, and you can pass any number of optimizers to your model when implementing it,
    :class:`pypots.clustering.crli.CRLI` for example.

    """

    def __init__(
        self,
        n_classes: int,
        batch_size: int,
        epochs: int,
        patience: Optional[int] = None,
        training_loss: Optional[Criterion] = CrossEntropy(),
        validation_metric: Optional[Criterion] = PR_AUC(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        enable_amp: bool = False,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            training_loss=training_loss,
            validation_metric=validation_metric,
            num_workers=num_workers,
            device=device,
            enable_amp=enable_amp,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        self.n_classes = n_classes

        # fetch the names of training loss and validation metric
        self.training_loss_name = self.training_loss.__class__.__name__
        self.validation_metric_name = self.validation_metric.__class__.__name__

    @abstractmethod
    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble the given data into a dictionary for training input.

        Parameters
        ----------
        data :
            Input data from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model training.
        """
        raise NotImplementedError

    @abstractmethod
    def _assemble_input_for_validating(self, data: list) -> dict:
        """Assemble the given data into a dictionary for validating input.

        Parameters
        ----------
        data :
            Data output from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model validating.
        """
        raise NotImplementedError

    @abstractmethod
    def _assemble_input_for_testing(self, data: list) -> dict:
        """Assemble the given data into a dictionary for testing input.

        Notes
        -----
        The processing functions of train/val/test stages are separated for the situation that the input of
        the three stages are different, and this situation usually happens when the Dataset/Dataloader classes
        used in the train/val/test stages are not the same, e.g. the training data and validating data in a
        classification task contains labels, but the testing data (from the production environment) generally
        doesn't have labels.

        Parameters
        ----------
        data :
            Data output from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model testing.
        """
        raise NotImplementedError

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    # model forward propagation processing
                    if os.getenv("ENABLE_AMP", False) and self.enable_amp:
                        with autocast():
                            self.optimizer.zero_grad()
                            results = self.model.forward(inputs)
                            results["loss"].sum().backward()  # sum() before backward() in case of multi-gpu training
                            self.optimizer.step()
                    else:
                        self.optimizer.zero_grad()
                        results = self.model.forward(inputs)
                        results["loss"].sum().backward()  # sum() before backward() in case of multi-gpu training
                        self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    epoch_val_pred_collector = []
                    epoch_val_label_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)

                            # model forward propagation processing
                            if os.getenv("ENABLE_AMP", False) and self.enable_amp:
                                with autocast():
                                    results = self.model.forward(inputs)
                            else:
                                results = self.model.forward(inputs)

                            epoch_val_pred_collector.append(results["classification_pred"])
                            epoch_val_label_collector.append(inputs["y"])

                    epoch_val_pred_collector = torch.cat(epoch_val_pred_collector, dim=-1)
                    epoch_val_label_collector = torch.cat(epoch_val_label_collector, dim=-1)

                    # TODO: refactor the following code to a function
                    epoch_val_pred_collector = np.argmax(epoch_val_pred_collector, axis=1)
                    mean_val_loss = self.validation_metric(epoch_val_pred_collector, epoch_val_label_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            self.validation_metric_name: mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss ({self.training_loss_name}): {mean_train_loss:.4f}, "
                        f"validation {self.validation_metric_name}: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"Epoch {epoch:03d} - training loss ({self.training_loss_name}): {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss:.4f}",
                )

                if os.getenv("ENABLE_HPO", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

    @abstractmethod
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the classifier on the given data.

        Parameters
        ----------
        train_set :
            The dataset for model training, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for training, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        val_set :
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if train_set and val_set are path strings.

        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def classify(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Classify the input data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples],
            Classification results of the given samples.
        """

        raise NotImplementedError
