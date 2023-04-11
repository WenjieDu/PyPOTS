"""
Base class for main models in PyPOTS.
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
    """Base model class for all model implementations.

    Parameters
    ----------
    device : str or `torch.device`, default = None,
        The device for the model to run on.
        If not given, will try to use CUDA devices first, then CPUs. CUDA and CPU are so far the main devices for people
        to train ML models. Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    tb_file_saving_path : str, default = None,
        The path to save the tensorboard file, which contains the loss values recorded during training.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        tb_file_saving_path: str = None,
    ):
        self.model = None

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

        # set up the summary writer for training log saving below
        if isinstance(tb_file_saving_path, str):

            from datetime import datetime

            # get the current time to append to the dir name,
            # so you can use the same tb_file_saving_path for multiple running
            time_now = datetime.now().__format__("%Y-%m-%d_T%H:%M:%S")
            # the actual directory name to save the tensorboard file
            actual_tb_saving_dir_name = "tensorboard_" + time_now
            actual_tb_file_saving_path = os.path.join(
                tb_file_saving_path, actual_tb_saving_dir_name
            )
            # os.makedirs(actual_tb_file_saving_path)  # create the dir for file saving
            self.summary_writer = SummaryWriter(
                actual_tb_file_saving_path, filename_suffix=".pypots"
            )
        else:
            # don't save the log if tb_file_saving_path isn't given, set summary_writer as None
            self.summary_writer = None

    def save_log_into_tb_file(self, step: int, stage: str, loss_dict: dict) -> None:
        """Saving training logs into the tensorboard file.

        Parameters
        ----------
        step : int,
            The current training step number.

        stage : str,
            The stage of the current operation, 'training' or 'validating'.

        loss_dict : dict,
            A dictionary containing items to log, should have at least one item, e.g. {'imputation loss': 0.05}

        """
        while len(loss_dict) > 0:
            (item_name, loss) = loss_dict.popitem()
            if "loss" in item_name:  # save all items containing word "loss" in the name
                self.summary_writer.add_scalar(f"{stage}/{item_name}", loss, step)

    def save_model(
        self,
        saving_dir: str,
        file_name: str,
        overwrite: bool = False,
    ) -> None:
        """Save the model to a disk file.

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
            logger.info(f"Saved successfully to {saving_path}.")
        except Exception as e:
            raise RuntimeError(f'{e} Failed to save the model to "{saving_path}"!')

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
    """Abstract class for all neural-network models.

    Parameters
    ----------
    batch_size : int,
        Size of the batch input into the model for one step.

    epochs : int,
        Training epochs, i.e. the maximum rounds of the model to be trained with.

    patience : int,
        Number of epochs the training procedure will keep if loss doesn't decrease.
        Once exceeding the number, the training will stop.

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

    tb_file_saving_path : str, default = None,
        The path to save the tensorboard file, which contains the loss values recorded during training.
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
        tb_file_saving_path: str = None,
    ):
        super().__init__(device, tb_file_saving_path)

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
        self.best_loss = float("inf")

    def _print_model_size(self) -> None:
        """Print the number of trainable parameters in the initialized NN model."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model initialized successfully with the number of trainable parameters: {num_params}"
        )
