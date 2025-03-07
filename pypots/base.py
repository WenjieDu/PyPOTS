"""
The base (abstract) classes for models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Optional, Union, Iterable, Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from .utils.file import create_dir_if_not_exist
from .utils.logging import logger, logger_creator


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
        2). assist users to optimize the model's hype-parameters.
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

        self.device = None  # set up with _setup_device() below
        self.enable_amp = enable_amp
        self.saving_path = None  # set up with _setup_path() below
        self.model_saving_strategy = model_saving_strategy
        self.verbose = verbose

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
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                logger.warning(
                    "‼️ You are trying to use AMP, but CUDA is not available in your environment. AMP will be disabled."
                )
            if not self.enable_amp:
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
        if isinstance(self.device, list):
            # parallely training on multiple devices
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device)
            self.model = self.model.cuda()
            logger.info(f"Model has been allocated to the given multiple devices: {self.device}")
        else:
            self.model = self.model.to(self.device)

    def _send_data_to_given_device(self, data) -> Iterable:
        if isinstance(self.device, (torch.device, list)):  # single device or parallely training on multiple devices
            if isinstance(self.device, list):
                data = map(lambda x: x.cuda(), data)
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
                    f"Use the arg `overwrite=True` to force overwrite."
                )
                return

        try:
            create_dir_if_not_exist(saving_dir)
            if isinstance(self.device, list):
                # to save a DataParallel model generically, save the model.module.state_dict()
                torch.save(self.model.module, saving_path)
            else:
                torch.save(self.model, saving_path)
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
            if isinstance(self.device, torch.device):
                loaded_model = torch.load(path, map_location=self.device)
            else:
                loaded_model = torch.load(path)
            if isinstance(loaded_model, torch.nn.Module):
                if isinstance(self.device, torch.device):
                    self.model.load_state_dict(loaded_model.state_dict())
                else:
                    self.model.module.load_state_dict(loaded_model.state_dict())
            else:
                self.model = loaded_model.model
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
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for training, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        val_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
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
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
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
        self.device = device
        self._send_model_to_given_device()


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

    train_loss_func:
        The customized loss function designed by users for training the model.
        If not given, will use the default loss as claimed in the original paper.

    val_metric_func:
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
        batch_size: int,
        epochs: int,
        patience: Optional[int] = None,
        train_loss_func: Optional[dict] = None,
        val_metric_func: Optional[dict] = None,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        enable_amp: bool = False,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            device,
            enable_amp,
            saving_path,
            model_saving_strategy,
            verbose,
        )

        # check patience
        if patience is None:
            patience = -1  # early stopping on patience won't work if it is set as < 0
        else:
            assert (
                patience <= epochs
            ), f"patience must be smaller than epochs which is {epochs}, but got patience={patience}"

        # check train_loss_func and val_metric_func
        train_loss_func_name, val_metric_func_name = "default", "loss (default)"
        if train_loss_func is not None:
            train_loss_func_name = train_loss_func.__class__.__name__
            assert isinstance(train_loss_func, Callable), "train_loss_func should be a callable instance"
            logger.info(f"Using customized {train_loss_func_name} as the training loss function.")
        if val_metric_func is not None:
            val_metric_func_name = val_metric_func.__class__.__name__
            assert isinstance(val_metric_func, Callable), "val_metric_func should be a callable instance"
            logger.info(f"Using customized {val_metric_func_name} as the validation metric function.")

        # set up the hype-parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.train_loss_func = train_loss_func
        self.train_loss_func_name = train_loss_func_name
        self.val_metric_func = val_metric_func
        self.val_metric_func_name = val_metric_func_name
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
    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        raise NotImplementedError
