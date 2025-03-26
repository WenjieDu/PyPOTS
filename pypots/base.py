"""
The base (abstract) classes for models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Optional, Union, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .nn.functional import autocast
from .nn.modules.loss import Criterion
from .utils.file import create_dir_if_not_exist
from .utils.logging import logger, logger_creator

try:
    import nni
except ImportError:
    pass


class BaseModel(ABC):
    """The base model class for all model implementations.

    Parameters
    ----------
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

    Attributes
    ----------
    model : object, default = None
        The underlying model or algorithm to finish the task.

    summary_writer : None or torch.utils.tensorboard.SummaryWriter,  default = None,
        The event writer to save training logs. Default as None. It only works when parameter `tb_file_saving_path` is
        given, otherwise the training events won't be saved.

        It is designed as being set up while initializing the model because it's created to
        1). help visualize the model's training procedure (during training not after) and
        2). assist users to optimize the model's hyperparameters.
        If only setting it up after training with a function like setter(), it cannot achieve the 1st purpose.

    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device, list]] = None,
        enable_amp: bool = False,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        saving_strategies = [None, "best", "better", "all"]
        assert (
            model_saving_strategy in saving_strategies
        ), f"saving_strategy must be one of {saving_strategies}, but got f{model_saving_strategy}."
        if saving_path is not None and saving_strategies is None:
            logger.warning("‼️ saving_path is given, but model_saving_strategy is None. No model file will be saved.")

        self.device = None  # set up with _setup_device() below
        self.saving_path = None  # set up with _setup_path() below
        self.model_saving_strategy = model_saving_strategy
        self.verbose = verbose

        # default as false, determine in _setup_device() with consideration on enable_amp and cuda availability
        self.amp_enabled = False
        self.enable_amp = enable_amp

        if not self.verbose:
            logger_creator.set_level("warning")

        self.model = None
        self.summary_writer = None

        # set up the device for model running below
        self._setup_device(device)

        # set up saving_path to save the trained model and training logs
        self._setup_path(saving_path)

    def _setup_device(self, device: Union[None, str, torch.device, list]) -> None:
        if device is None:
            # if it is None, then use the first cuda device if cuda is available, otherwise use cpu
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            logger.info(f"No given device, using default device: {self.device}")
        else:
            if isinstance(device, str):
                self.device = torch.device(device.lower())
            elif isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, list):
                if len(device) == 0:
                    raise ValueError("The list of devices should have at least 1 device, but got 0.")
                elif len(device) == 1:
                    return self._setup_device(device[0])
                # parallely training on multiple CUDA devices

                # ensure the list is not empty
                device_list = []
                for idx, d in enumerate(device):
                    if isinstance(d, str):
                        d = d.lower()
                        assert (
                            "cuda" in d
                        ), "The feature of training on multiple devices currently only support CUDA devices."
                        device_list.append(torch.device(d))
                    elif isinstance(d, torch.device):
                        assert (
                            "cuda" in d.type
                        ), "The feature of training on multiple devices currently only support CUDA devices."
                        device_list.append(d)
                    else:
                        raise TypeError(
                            f"Devices in the list should be str or torch.device, "
                            f"but the device with index {idx} is {type(d)}."
                        )
                if len(device_list) > 1:
                    self.device = device_list
                else:
                    self.device = device_list[0]
            else:
                raise TypeError(
                    f"device should be str/torch.device/a list containing str or torch.device, but got {type(device)}"
                )

            logger.info(f"Using the given device: {self.device}")

        # check CUDA availability if using CUDA
        if (isinstance(self.device, list) and "cuda" in self.device[0].type) or (
            isinstance(self.device, torch.device) and "cuda" in self.device.type
        ):
            assert (
                torch.cuda.is_available() and torch.cuda.device_count() > 0
            ), "You are trying to use CUDA for model training, but CUDA is not available in your environment."

        if os.getenv("ENABLE_AMP", False):
            if self.enable_amp:
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    logger.warning(
                        "‼️ You are trying to use AMP, but CUDA is not available in your environment. "
                        "AMP will be disabled."
                    )
                else:
                    self.amp_enabled = True
            else:
                logger.warning(
                    f"‼️ You are trying to use AMP, but the model {self.__class__.__name__} "
                    "does not support AMP operation. AMP will be disabled."
                )

    def _setup_path(self, saving_path) -> None:
        MODEL_NO_NEED_TO_SAVE = [
            "LOCF",
            "Median",
            "Mean",
        ]
        # if the model is no need to save (e.g. LOCF), then skip the following steps
        if self.__class__.__name__ in MODEL_NO_NEED_TO_SAVE:
            return

        if isinstance(saving_path, str):
            # get the current time to append to saving_path,
            # so you can use the same saving_path to run multiple times
            # and also be aware of when they were run
            time_now = datetime.now().__format__("%Y%m%d_T%H%M%S")
            # the actual saving_path for saving both the best model and the tensorboard file
            self.saving_path = os.path.join(saving_path, time_now)

            # initialize self.summary_writer only if saving_path is given and not None
            # otherwise self.summary_writer will be None and the training log won't be saved
            tb_saving_path = os.path.join(self.saving_path, "tensorboard")
            self.summary_writer = SummaryWriter(
                tb_saving_path,
                filename_suffix=".pypots",
            )
            logger.info(f"Model files will be saved to {self.saving_path}")
            logger.info(f"Tensorboard file will be saved to {tb_saving_path}")
        else:
            logger.warning("‼️ saving_path not given. Model files and tensorboard file will not be saved.")

    def _send_model_to_given_device(self) -> None:
        if isinstance(self.model, torch.nn.DataParallel):
            # in this case, the model has been sent to multi-gpu previously,
            # and we have to turn the model from nn.DataParallel to nn.Module first
            self.model = self.model.module

        if isinstance(self.device, list):
            # parallely training on multiple devices
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
            self.model = self.model.to(self.device[0])
            logger.info(f"Model has been allocated to the given multiple devices: {self.device}")
        else:
            self.model = self.model.to(self.device)

    def _send_data_to_given_device(self, data) -> Iterable:
        if isinstance(self.device, (torch.device, list)):  # single device or parallely training on multiple devices
            if isinstance(self.device, list):
                data = map(lambda x: x.to(self.device[0]), data)
            else:
                data = map(lambda x: x.to(self.device), data)

        else:  # CPU
            data = map(lambda x: x.to("cpu"), data)

        return data

    def _save_log_into_tb_file(self, step: int, stage: str, loss_dict: dict) -> None:
        """Saving training logs into the tensorboard file specified by the given path `tb_file_saving_path`.

        Parameters
        ----------
        step :
            The current training step number.
            One step for one batch processing, so the number of steps means how many batches the model has processed.

        stage :
            The stage of the current operation, e.g. 'pretraining', 'training', 'validating'.

        loss_dict :
            A dictionary containing items to log, should have at least one item, and only items having its name
            including "loss" or "error" will be logged, e.g. {'imputation_loss': 0.05, "classification_error": 0.32}.

        """
        while len(loss_dict) > 0:
            (item_name, loss) = loss_dict.popitem()
            # save all items containing "loss" or "error" in the name
            # WDU: may enable customization keywords in the future
            if ("loss" in item_name) or ("error" in item_name):
                if isinstance(loss, torch.Tensor):
                    loss = loss.sum()
                self.summary_writer.add_scalar(f"{stage}/{item_name}", loss, step)

    def _auto_save_model_if_necessary(
        self,
        confirm_saving: bool = True,
        saving_name: str = None,
    ) -> None:
        """Automatically save the current model into a file if in need.

        Parameters
        ----------
        confirm_saving :
            One more condition to confirm saving the model.

        saving_name :
            The file name of the saved model.

        """
        if self.saving_path is not None and self.model_saving_strategy is not None:
            # construct the saving path
            name = self.__class__.__name__ if saving_name is None else saving_name
            saving_path = os.path.join(self.saving_path, name)

            if self.model_saving_strategy == "all":
                self.save(saving_path)
            elif self.model_saving_strategy == "better" and confirm_saving:
                self.save(saving_path)
            elif self.model_saving_strategy == "best" and confirm_saving:
                self.save(saving_path)
            else:
                pass

    def _organize_content_to_save(self):
        from .version import __version__ as pypots_version

        # all_attrs = self.__dict__
        # del all_attrs["model"]

        if isinstance(self.device, list):
            # to save a DataParallel model generically, save the model.module.state_dict()
            model_state_dict = deepcopy(self.model.module.state_dict())
        else:
            model_state_dict = deepcopy(self.model.state_dict())

        all_attrs = dict({})
        all_attrs["model_state_dict"] = model_state_dict
        all_attrs["pypots_version"] = pypots_version

        return all_attrs

    def save(
        self,
        saving_path: str,
        overwrite: bool = False,
    ) -> None:
        """Save the model with current parameters to a disk file.

        A ``.pypots`` extension will be appended to the filename if it does not already have one.
        Please note that such an extension is not necessary, but to indicate the saved model is from PyPOTS framework
        so people can distinguish.

        Parameters
        ----------
        saving_path :
            The given path to save the model. The directory will be created if it does not exist.

        overwrite :
            Whether to overwrite the model file if the path already exists.

        """
        # split the saving dir and file name from the given path
        saving_dir, file_name = os.path.split(saving_path)
        # if parent dir is not given, save in the current dir
        saving_dir = "." if saving_dir == "" else saving_dir
        # add the suffix ".pypots" if not given
        if file_name.split(".")[-1] != "pypots":
            file_name += ".pypots"
        # rejoin the path for saving the model
        saving_path = os.path.join(saving_dir, file_name)

        if os.path.exists(saving_path):
            if overwrite:
                logger.warning(f"‼️ File {saving_path} exists. Argument `overwrite` is True. Overwriting now...")
            else:
                logger.error(
                    f"❌ File {saving_path} exists. Saving operation aborted. "
                    "Use the arg `overwrite=True` to force overwrite."
                )
                return

        try:
            create_dir_if_not_exist(saving_dir)
            content_to_save = self._organize_content_to_save()
            torch.save(content_to_save, saving_path)
            logger.info(f"Saved the model to {saving_path}")

        except Exception as e:
            raise RuntimeError(f'Failed to save the model to "{saving_path}" because of the below error! \n{e}')

    def load(self, path: str) -> None:
        """Load the saved model from a disk file.

        Parameters
        ----------
        path :
            The local path to a disk file saving the trained model.

        Notes
        -----
        If the training environment and the deploying/test environment use the same type of device (GPU/CPU),
        you can load the model directly with torch.load(model_path).

        """
        assert os.path.exists(path), f"Model file {path} does not exist."

        try:
            map_location = self.device[0] if isinstance(self.device, list) else self.device
            loaded_file = torch.load(path, map_location=map_location)

            if isinstance(loaded_file, torch.nn.Module):  # compatible model for pypots <0.13
                if isinstance(self.device, torch.device):
                    self.model.load_state_dict(loaded_file.state_dict())
                else:
                    self.model.module.load_state_dict(loaded_file.state_dict())
                logger.warning(
                    "‼️ This model file is saved with pypots <0.13 and "
                    "has been loaded with the compatible mode which will be deprecated in the future. "
                    "Please save the model again with the later versions (>=0.13) of PyPOTS and "
                    "delete the old model file."
                )
            else:  # loading strategy for pypots >=0.13
                loaded_model_dict = loaded_file["model_state_dict"]

                if isinstance(self.device, torch.device):
                    current_model_dict = self.model.state_dict()
                    current_model_dict.update(loaded_model_dict)
                    self.model.load_state_dict(current_model_dict)
                else:
                    current_model_dict = self.model.module.state_dict()
                    current_model_dict.update(loaded_model_dict)
                    self.model.module.load_state_dict(current_model_dict)

            self.model.eval()  # set the model as eval status to freeze it.

        except Exception as e:
            raise e
        logger.info(f"Model loaded successfully from {path}")

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
            The dataset for model training, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for training, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        val_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
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
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like with shape [n_samples, n_steps, n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        Returns
        -------
        result_dict :
            Prediction results in a Python Dictionary for the given samples.
            It should be a dictionary including keys as 'imputation', 'classification', 'clustering', and 'forecasting'.
            For sure, only the keys that relevant tasks are supported by the model will be returned.
        """
        raise NotImplementedError

    def to(self, device: Union[str, torch.device]) -> None:
        """Move the model to the given device.

        Parameters
        ----------
        device :
            The device to move the model to. It can be a string or a :class:`torch.device` object.

        """
        self._setup_device(device)
        self._send_model_to_given_device()
        # TODO: have to move the optimizer to the given device as well
        #  but we may have multi optimizers for a model, e.g. GANs, https://github.com/WenjieDu/PyPOTS/issues/599


class BaseNNModel(BaseModel):
    """The abstract class for all neural-network models.

    Parameters
    ----------
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
        If not given, the model will be trained with its own loss defined in its paper and fixed in the implementation.

    validation_metric:
        The customized metric function designed by users for validating the model.
        If not given, the model's training loss will be used as the validation metric to select the best model.

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

    Attributes
    ---------
    best_model_dict : dict, default = None,
        A dictionary contains the trained model that achieves the best performance according to the loss defined,
        i.e. the lowest loss.

    best_loss : float, default = inf,
        The criteria to judge whether the model's performance is the best so far.
        Usually the lower, the better.

    best_epoch : int, default = -1,
        The epoch number when the best loss is got.

    Notes
    -----
    Optimizers are necessary for training deep-learning neural networks, but we don't put a parameter ``optimizer``
    here because some models (e.g. GANs) need more than one optimizer (e.g. one for generator, one for discriminator),
    and ``optimizer`` is ambiguous for them. Therefore, we leave optimizers as parameters for concrete model
    implementations, and you can pass any number of optimizers to your model when implementing it,
    :class:`pypots.clustering.crli.CRLI` for example.

    """

    def __init__(
        self,
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
            device=device,
            enable_amp=enable_amp,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        # check patience
        if patience is None:
            patience = -1  # early stopping on patience won't work if it is set as < 0
        else:
            assert (
                patience <= epochs
            ), f"patience must be smaller than epochs which is {epochs}, but got patience={patience}"

        # check training_loss and validation_metric
        training_loss_name, validation_metric_name = "default", "loss"  # default names for loss and metric
        # determine the training_loss and training_loss_name
        if not isinstance(training_loss, Criterion):  # if training_loss is a class, instantiate it
            training_loss = training_loss()
            assert isinstance(training_loss, Criterion)
        if training_loss.__class__.__name__ == "Criterion":
            # in this case, we may need self.training_loss.lower_better.
            # In addition, training_loss won't be invoked and the model will be trained with its own loss
            # defined in its paper and fixed in the implementation
            pass
        else:
            training_loss_name = training_loss.__class__.__name__
            logger.info(f"Using customized {training_loss_name} as the training loss function.")
        # determine the validation_metric and validation_metric_name
        if not isinstance(validation_metric, Criterion):  # if validation_metric is a class, instantiate it
            validation_metric = validation_metric()
            assert isinstance(validation_metric, Criterion)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need self.validation_metric.lower_better in _train_model()
            # In addition, validation_metric won't be invoked and the model's training loss will be used as
            # the validation metric to select the best model
            pass
        else:
            validation_metric_name = validation_metric.__class__.__name__
            logger.info(f"Using customized {validation_metric_name} as the validation metric function.")

        # set up the hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.training_loss = training_loss
        self.training_loss_name = training_loss_name
        self.validation_metric = validation_metric
        self.validation_metric_name = validation_metric_name
        self.original_patience = patience
        self.num_workers = num_workers

        self.model = None
        self.num_params = None
        self.optimizer = None
        self.best_model_dict = None
        self.best_loss = float("inf")
        self.best_epoch = -1

    def _print_model_size(self) -> None:
        """Print the number of trainable parameters in the initialized NN model."""
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"{self.__class__.__name__} initialized with the given hyperparameters, "
            f"the number of trainable parameters: {self.num_params:,}"
        )

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
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_model_dict = None

        if self.validation_metric.lower_better:
            self.best_loss = float("inf")
        else:
            self.best_loss = float("-inf")

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(train_dataloader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)

                    with autocast(enabled=self.amp_enabled):
                        self.optimizer.zero_grad()
                        results = self.model(inputs, calc_criterion=True)
                        loss = results["loss"].sum()
                        loss.backward()
                        self.optimizer.step()
                    epoch_train_loss_collector.append(loss.item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)
                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_dataloader is not None:
                    self.model.eval()
                    val_metric_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_dataloader):
                            inputs = self._assemble_input_for_validating(data)

                            with autocast(enabled=self.amp_enabled):
                                results = self.model(inputs, calc_criterion=True)

                            val_metric = results["metric"].sum()
                            val_metric_collector.append(val_metric.detach().item())

                    mean_val_metric = np.mean(val_metric_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_metric_dict = {
                            self.validation_metric_name: mean_val_metric,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_metric_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss ({self.training_loss_name}): {mean_train_loss:.4f}, "
                        f"validation {self.validation_metric_name}: {mean_val_metric:.4f}"
                    )
                    mean_loss = mean_val_metric
                else:
                    logger.info(f"Epoch {epoch:03d} - training loss ({self.training_loss_name}): {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Got NaN loss in epoch#{epoch}. This may lead to unexpected errors.")

                if (self.validation_metric.lower_better and mean_loss < self.best_loss) or (
                    not self.validation_metric.lower_better and mean_loss > self.best_loss
                ):
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_{self.validation_metric_name}{mean_loss:.4f}",
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

        if np.isnan(self.best_loss) or self.best_loss.__eq__(float("inf")):
            raise ValueError("Something is wrong. best_loss is NaN/Inf after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

    @abstractmethod
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        raise NotImplementedError
