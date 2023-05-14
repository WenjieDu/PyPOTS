"""
The base class for PyPOTS imputation models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import os
from abc import abstractmethod
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from pypots.base import BaseModel, BaseNNModel
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae

try:
    import nni
except ImportError:
    pass


class BaseImputer(BaseModel):
    """Abstract class for all imputation models.

    Parameters
    ----------
    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first, then CPUs. CUDA and CPU are so far the main devices for people
        to train ML models. Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, default = None,
        The path to save model checkpoints and tensorboard files,
        which contains the loss values recorded during training.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            device,
            saving_path,
            model_saving_strategy,
        )

    @abstractmethod
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
        pass

    @abstractmethod
    def impute(
        self,
        X: Union[dict, str],
        file_type: str = "h5py",
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
        pass


class BaseNNImputer(BaseNNModel, BaseImputer):
    """Abstract class for all neural-network imputation models.

    Parameters
    ----------
    batch_size : int,
        Size of the batch input into the model for one step.

    epochs : int,
        Training epochs, i.e. the maximum rounds of the model to be trained with.

    patience : int,
        Number of epochs the training procedure will keep if loss doesn't decrease.
        Once exceeding the number, the training will stop.

    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first, then CPUs. CUDA and CPU are so far the main devices for people
        to train ML models. Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, default = None,
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    """

    def __init__(
        self,
        batch_size: int,
        epochs: int,
        patience: int,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
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

    @abstractmethod
    def _assemble_input_for_training(self, data: list) -> dict:
        """Assemble the given data into a dictionary for training input.

        Parameters
        ----------
        data : list,
            Input data from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model training.
        """
        pass

    @abstractmethod
    def _assemble_input_for_validating(self, data: list) -> dict:
        """Assemble the given data into a dictionary for validating input.

        Parameters
        ----------
        data : list,
            Data output from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model validating.
        """
        pass

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
        data : list,
            Data output from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model testing.
        """
        pass

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
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)
                    self.optimizer.zero_grad()
                    results = self.model.forward(inputs)
                    results["loss"].backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    imputation_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            imputed_data = self.model.impute(inputs)
                            imputation_collector.append(imputed_data)

                    imputation_collector = torch.cat(imputation_collector)
                    imputation_collector = imputation_collector.cpu().detach().numpy()

                    mean_val_loss = cal_mae(
                        imputation_collector,
                        val_loader.dataset.data["X_intact"],
                        val_loader.dataset.data["indicating_mask"],
                        # the above val_loader.dataset.data is a dict containing the validation dataset
                    )

                    # save validating loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"epoch {epoch}: "
                        f"training loss {mean_train_loss:.4f}, "
                        f"validating loss {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"epoch {epoch}: training loss {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                    # save the model if necessary
                    self._auto_save_model_if_necessary(
                        training_finished=False,
                        saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss}",
                    )
                else:
                    self.patience -= 1

                if os.getenv("enable_nni", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info(
                        "Exceeded the training patience. Terminating the training procedure..."
                    )
                    break

        except Exception as e:
            logger.info(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not get trained. Please try fit() again."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. "
                    "Model will load the best parameters so far for testing. "
                    "If you don't want it, please try fit() again."
                )

        if np.equal(self.best_loss.item(), float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info("Finished training.")
