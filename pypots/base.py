"""
Base class for main models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter


class BaseModel(ABC):
    """ Base class for all models.
    """

    def __init__(self, device):
        self.logger = {}

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def save_logs_to_tensorboard(self, saving_path):
        tb_summary_writer = SummaryWriter(saving_path)
        tb_summary_writer.add_custom_scalars(self.logger)
        tb_summary_writer.close()

    def save_model(self, saving_path):
        """ Save the model to a disk file.

        Parameters
        ----------
        saving_path : str,
            The given path to save the model.

        Notes
        -----
        Use model=torch.load(path_to_saved_model) to load the saved model.
        """
        try:
            torch.save(self, saving_path)
        except Exception as e:
            print(e)
        print(f'Saved successfully to {saving_path}.')
