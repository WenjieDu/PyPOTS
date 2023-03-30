"""
Base class for main models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import os
from abc import ABC

import torch

from pypots.utils.files import create_dir_if_not_exist
from pypots.utils.logging import logger


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(self, device):
        self.logger = {}
        self.model = None

        if device is None:
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
            logger.info(f"No given device, using default device: {self.device}")
        else:
            self.device = device

    def save_logs_to_tensorboard(self, saving_path):
        """Save logs (self.logger) into a tensorboard file.

        Parameters
        ----------
        saving_path : str
            Local disk path to save the tensorboard file.
        """
        # TODO: find a solution for log saving
        raise IOError("This function is not ready for users.")
        # tb_summary_writer = SummaryWriter(saving_path)
        # tb_summary_writer.add_custom_scalars(self.logger)
        # tb_summary_writer.close()
        # logger.info(f'Log saved successfully to {saving_path}.')

    def save_model(self, saving_dir, name, overwrite=False):
        """Save the model to a disk file.

        A .pypots extension will be appended to the filename if it does not already have one.
        Please note that such an extension is not necessary, but to indicate the saved model is from PyPOTS framework so people can distinguish.

        Parameters
        ----------
        saving_dir : str,
            The given directory to save the model.

        name : str,
            The file name of the model to be saved.

        overwrite : bool,

        """
        name = name + ".pypots" if name.split(".")[-1] != "pypots" else name
        saving_path = os.path.join(saving_dir, name)
        if os.path.exists(saving_path):
            if overwrite:
                logger.warning(
                    f"File {saving_path} exists. Argument `overwrite` is True. Overwriting now..."
                )
            else:
                logger.error(f"File {saving_path} exists. Saving operation aborted.")
                return
        try:
            create_dir_if_not_exist(saving_dir)
            torch.save(self.model, saving_path)
            logger.info(f"Saved successfully to {saving_path}.")
        except Exception as e:
            raise RuntimeError(f'{e} Failed to save the model to "{saving_path}"!')

    def load_model(self, model_path):
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
    """Abstract class for all neural-network models."""

    def __init__(
        self, learning_rate, epochs, patience, batch_size, weight_decay, device
    ):
        super().__init__(device)

        # training hype-parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.original_patience = patience
        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.model = None
        self.optimizer = None
        self.best_model_dict = None
        self.best_loss = float("inf")
        self.logger = {"training_loss": [], "validating_loss": []}

    def _print_model_size(self):
        """Print the number of trainable parameters in the initialized NN model."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model initialized successfully. Number of the trainable parameters: {num_params}"
        )
