"""
Base class for main models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3
from abc import ABC

import torch
from torch.utils.tensorboard import SummaryWriter


class BaseModel(ABC):
    """ Base class for all models.
    """

    def __init__(self, device):
        self.logger = {}
        self.model = None

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
        """
        try:
            torch.save(self.model, saving_path)
        except Exception as e:
            print(e)
        print(f'Saved successfully to {saving_path}.')

    def load_model(self, model_path):
        """ Load the saved model from a disk file.

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
            if isinstance(loaded_model.model, torch.nn.Module):
                self.model.load_state_dict(loaded_model.model.state_dict())
            else:
                self.model = loaded_model.model
        except Exception as e:
            raise e
        print(f'Model loaded successfully from {model_path}.')


class BaseNNModel(BaseModel):
    """ Abstract class for all neural-network models.
    """

    def __init__(self, seq_len, n_features, learning_rate, epochs, patience, batch_size, weight_decay, device):
        super(BaseNNModel, self).__init__(device)

        self.seq_len = seq_len
        self.n_features = n_features

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
        self.best_loss = float('inf')
        self.logger = {
            'training_loss': [],
            'validating_loss': []
        }
