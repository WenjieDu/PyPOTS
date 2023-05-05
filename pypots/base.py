"""
The base (abstract) classes for models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
from abc import ABC
from typing import Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from pypots.utils.files import create_dir_if_not_exist
from pypots.utils.logging import logger


class BaseModel(ABC):
    """The base model class for all model implementations.

    Parameters
    ----------
    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first (will use the GPU with device number 0 only by default),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, default = None,
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str or None, None or "best" or "better" , default = "best",
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    Attributes
    ----------
    model : object, default = None,
        The underlying model or algorithm to finish the task.

    summary_writer : None or torch.utils.tensorboard.SummaryWriter,  default = None,
        The event writer to save training logs. Default as None. It only works when parameter `tb_file_saving_path` is
        given, otherwise the training events won't be saved.

        It is designed as being set up while initializing the model because it's created to
        1). help visualize the model's training procedure (during training not after) and
        2). assist users to tune the model's hype-parameters.
        If only setting it up after training with a function like setter(), it cannot achieve the 1st purpose.

    """

    # leverage typing to show type hints in IDEs
    # SAVING_STRATEGY = Literal["best", "better"]
    SAVING_STRATEGY = [None, "best", "better"]

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):

        assert model_saving_strategy in [
            None,
            "best",
            "better",
        ], f"saving_strategy must be one of {self.SAVING_STRATEGY}, but got f{model_saving_strategy}."

        self.device = None
        self.saving_path = saving_path
        self.model_saving_strategy = model_saving_strategy

        self.model = None
        self.summary_writer = None

        # set up the device for model running below
        if device is None:
            # if it is None, then
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
            logger.info(f"No given device, using default device: {self.device}")
        else:
            if isinstance(device, str):
                self.device = torch.device(device)
            elif isinstance(device, torch.device):
                self.device = device
            else:
                raise TypeError(
                    f"device should be str or torch.device, but got {type(device)}"
                )

        # set up saving_path to save the trained model and training logs
        if isinstance(saving_path, str):
            from datetime import datetime

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

            logger.info(
                f"the trained model will be saved to {self.saving_path}, "
                f"the tensorboard file will be saved to {tb_saving_path}"
            )

    def save_log_into_tb_file(self, step: int, stage: str, loss_dict: dict) -> None:
        """Saving training logs into the tensorboard file specified by the given path `tb_file_saving_path`.

        Parameters
        ----------
        step : int,
            The current training step number.
            One step for one batch processing, so the number of steps means how many batches the model has processed.

        stage : str,
            The stage of the current operation, e.g. 'pretraining', 'training', 'validating'.

        loss_dict : dict,
            A dictionary containing items to log, should have at least one item, and only items having its name
            including "loss" or "error" will be logged, e.g. {'imputation_loss': 0.05, "classification_error": 0.32}.

        """
        while len(loss_dict) > 0:
            (item_name, loss) = loss_dict.popitem()
            # save all items containing "loss" or "error" in the name
            # WDU: may enable customization keywords in the future
            if ("loss" in item_name) or ("error" in item_name):
                self.summary_writer.add_scalar(f"{stage}/{item_name}", loss, step)

    def save_model(
        self,
        saving_dir: str,
        file_name: str,
        overwrite: bool = False,
    ) -> None:
        """Save the model with current parameters to a disk file.

        A .pypots extension will be appended to the filename if it does not already have one.
        Please note that such an extension is not necessary, but to indicate the saved model is from PyPOTS framework
        so people can distinguish.

        Parameters
        ----------
        saving_dir : str,
            The given directory to save the model.

        file_name : str,
            The file name of the model to be saved.

        overwrite : bool, default = False,
            Whether to overwrite the model file if the path already exists.

        """
        file_name = (
            file_name + ".pypots" if file_name.split(".")[-1] != "pypots" else file_name
        )
        saving_path = os.path.join(saving_dir, file_name)

        if os.path.exists(saving_path):
            if overwrite:
                logger.warning(
                    f"File {saving_path} exists. Argument `overwrite` is True. Overwriting now..."
                )
            else:
                logger.error(f"File {saving_path} exists. Saving operation aborted.")
        try:
            create_dir_if_not_exist(saving_dir)
            torch.save(self.model, saving_path)
            logger.info(f"Saved the model to {saving_path}.")
        except Exception as e:
            raise RuntimeError(
                f'Failed to save the model to "{saving_path}" because of the below error! \n{e}'
            )

    def auto_save_model_if_necessary(
        self,
        training_finished: bool = True,
        saving_name: str = None,
    ):
        """Automatically save the current model into a file if in need.

        Parameters
        ----------
        training_finished : bool, default = False,
            Whether the training is already finished when invoke this function.
            The saving_strategy "better" only works when training_finished is False.
            The saving_strategy "best" only works when training_finished is True.

        saving_name : str, default = None,
            The file name of the saved model.

        """
        if self.saving_path is not None and self.model_saving_strategy is not None:
            name = self.__class__.__name__ if saving_name is None else saving_name
            if not training_finished and self.model_saving_strategy == "better":
                self.save_model(self.saving_path, name)
            elif training_finished and self.model_saving_strategy == "best":
                self.save_model(self.saving_path, name)
        else:
            return

    def load_model(self, model_path: str) -> None:
        """Load the saved model from a disk file.

        Parameters
        ----------
        model_path : str,
            Local path to a disk file saving trained model.

        Notes
        -----
        If the training environment and the deploying/test environment use the same type of device (GPU/CPU),
        you can load the model directly with torch.load(model_path).

        """
        try:
            loaded_model = torch.load(model_path, map_location=self.device)
            if isinstance(loaded_model, torch.nn.Module):
                self.model.load_state_dict(loaded_model.state_dict())
            else:
                self.model = loaded_model.model
        except Exception as e:
            raise e
        logger.info(f"Model loaded successfully from {model_path}.")


class BaseNNModel(BaseModel):
    """The abstract class for all neural-network models.

    Parameters
    ----------
    batch_size : int,
        Size of the batch input into the model for one step.

    epochs : int,
        Training epochs, i.e. the maximum rounds of the model to be trained with.

    patience : int,
        Number of epochs the training procedure will keep if loss doesn't decrease.
        Once exceeding the number, the training will stop.
        Must be smaller than or equal to the value of `epoches`.

    learning_rate : float,
        The learning rate of the optimizer.

    weight_decay : float,
        The weight decay of the optimizer.

    num_workers : int, default = 0,
            The number of subprocesses to use for data loading.
            `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first, then CPUs. CUDA and CPU are so far the main devices for people
        to train ML models. Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, default = None,
        The path to save the tensorboard file, which contains the loss values recorded during training.

    model_saving_strategy : str or None, None or "best" or "better" , default = "best",
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.


    Attributes
    ---------
    optimizer : torch.optim.Optimizer, default = None,
        The optimizer to back propagate losses for model optimization. Default as None, will be implemented
        when the concreate implementation model gets initialized.

    best_model_dict : dict, default = None,
        A dictionary contains the trained model that achieves the best performance according to the loss defined,
        i.e. the lowest loss.

    best_loss : float, default = inf,
        The criteria to judge whether the model's performance is the best so far.
        Usually the lower, the better.
    """

    def __init__(
        self,
        batch_size: int,
        epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(device, saving_path, model_saving_strategy)

        if patience is None:
            patience = -1  # early stopping on patience won't work if it is set as < 0
        else:
            assert (
                patience <= epochs
            ), f"patience must be smaller than epoches which is {epochs}, but got patience={patience}"

        # training hype-parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.original_patience = patience
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers

        self.model = None
        self.optimizer = None
        self.best_model_dict = None
        # WDU: may enable users to customize the criteria in the future
        self.best_loss = float("inf")

    def _print_model_size(self) -> None:
        """Print the number of trainable parameters in the initialized NN model."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model initialized successfully with the number of trainable parameters: {num_params:,}"
        )
