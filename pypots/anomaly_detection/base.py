"""
The base class for PyPOTS anomaly detection models.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from abc import abstractmethod
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..base import BaseModel, BaseNNModel
from ..data.dataset.base import BaseDataset
from ..nn.functional import autocast
from ..nn.modules.loss import Criterion
from ..utils.logging import logger


class BaseDetector(BaseModel):
    """Abstract class for all anomaly detection models.

    Parameters
    ----------
    anomaly_rate :
        The rate of anomalies in the data, should be in the range (0, 1).

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
    """

    def __init__(
        self,
        anomaly_rate,
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

        assert 0 < anomaly_rate < 1, f"anomaly_rate should be in (0, 1), but got {anomaly_rate}"
        self.anomaly_rate = anomaly_rate

    @abstractmethod
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the detector on the given data.

        Parameters
        ----------
        train_set :
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        val_set :
            The dataset for model validating, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for validating, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type :
            The type of the given file if train_set and val_set are path strings.

        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The test dataset for model to process, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is the time-series data for processing.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include 'X' key.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        result_dict :
            The dictionary containing the anomaly detection results as key 'anomaly_detection'
            and latent variables if necessary.

        """
        raise NotImplementedError

    def detect(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> np.ndarray:
        """Detect anomalies in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like with shape [n_samples, n_steps, n_features], or a path
            string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, with shape [n_samples, n_steps, n_features],
            Anomaly detection results in the input data.
        """
        result_dict = self.predict(
            test_set,
            file_type,
            **kwargs,
        )
        results = result_dict["anomaly_detection"]
        return results


class BaseNNDetector(BaseNNModel):
    """The abstract class for all neural-network anomaly detection models in PyPOTS.

    Parameters
    ----------
    anomaly_rate :
        The rate of anomalies in the data, should be in the range (0, 1).

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
        If not given, will use the default MSE metric.

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
        anomaly_rate: float,
        training_loss: Union[Criterion, type],
        validation_metric: Union[Criterion, type],
        batch_size: int,
        epochs: int,
        patience: Optional[int] = None,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        enable_amp: bool = False,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            enable_amp=enable_amp,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        assert 0 < anomaly_rate < 1, f"anomaly_rate should be in (0, 1), but got {anomaly_rate}"
        self.anomaly_rate = anomaly_rate

        # we need train_set in predict(), hence init here and set at the beginning of fit()
        self.train_set = None

    @abstractmethod
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """Train the detector on the given data.

        Parameters
        ----------
        train_set :
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        val_set :
            The dataset for model validating, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for validating, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type :
            The type of the given file if train_set and val_set are path strings.

        """
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The test dataset for model to process, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is the time-series data for processing.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include 'X' key.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        result_dict :
            The dictionary containing the anomaly detection results as key 'anomaly_detection'
            and latent variables if necessary.

        """
        self.model.eval()  # set the model to evaluation mode

        # Step 1: wrap the input data with classes Dataset and DataLoader
        train_dataset = BaseDataset(
            self.train_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_dataset = BaseDataset(
            test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Step 2: process the data with the model
        score_collector = []
        anomaly_criterion = torch.nn.MSELoss(reduce=False)
        for idx, data in enumerate(train_dataloader):
            inputs = self._assemble_input_for_testing(data)
            with autocast(enabled=self.amp_enabled):
                results = self.model(inputs, **kwargs)
            outputs = results["reconstruction"]
            score = torch.mean(anomaly_criterion(inputs["X"], outputs), dim=-1)
            score = score.detach().cpu().numpy()
            score_collector.append(score)

        attens_energy = np.concatenate(score_collector, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # find the threshold
        score_collector = []
        for idx, data in enumerate(test_dataloader):
            inputs = self._assemble_input_for_testing(data)
            with autocast(enabled=self.amp_enabled):
                results = self.model(inputs, **kwargs)
            outputs = results["reconstruction"]
            # criterion
            score = torch.mean(anomaly_criterion(inputs["X"], outputs), dim=-1)
            score = score.detach().cpu().numpy()
            score_collector.append(score)

        attens_energy = np.concatenate(score_collector, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.anomaly_rate * 100)
        logger.info(f"Threshold {threshold}")

        anomaly_pred = (test_energy > threshold).astype(int)

        result_dict = {
            "anomaly_detection": anomaly_pred,
        }
        return result_dict

    def detect(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> np.ndarray:
        """Detect anomalies in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like with shape [n_samples, n_steps, n_features], or a path
            string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        results :
            Anomaly detection results of the given samples.
        """
        result_dict = self.predict(
            test_set,
            file_type,
            **kwargs,
        )
        results = result_dict["anomaly_detection"]
        return results
