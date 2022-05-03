"""
Base class for main models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3
from abc import ABC

import numpy as np
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

    def check_input(self, expected_n_steps, expected_n_features, X, y=None):
        """ Check value type and shape of input X and y

        Parameters
        ----------
        expected_n_steps : int
            Number of time steps of input time series (X) that the model expects.
            This value is the same with the argument `n_steps` used to initialize the model.

        expected_n_features : int
            Number of feature dimensions of input time series (X) that the model expects.
            This value is the same with the argument `n_features` used to initialize the model.

        X : array-like,
            Time-series data that must have a shape like [n_samples, expected_n_steps, expected_n_features].

        y : array-like
            Labels of time-series samples (X) that must have a shape like [n_samples] or [n_samples, n_classes].

        Returns
        -------
        X : tensor

        y : tensor
        """
        if isinstance(X, torch.Tensor):
            pass
        elif isinstance(X, list):
            X = torch.tensor(X)
        elif isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        else:
            raise TypeError('X should be an instance of list/np.ndarray/torch.Tensor, '
                            f'but got {type(X)}')

        X_shape = X.shape
        assert len(X_shape) == 3, f'input should have 3 dimensions [n_samples, seq_len, n_features],' \
                                  f'but got shape={X.shape}'
        assert X_shape[1] == expected_n_steps
        assert X_shape[2] == expected_n_features

        if y is not None:
            assert len(X) == len(y), f'lengths of X and y must match, ' \
                                     f'but got f{len(X)} and {len(y)}'
            if isinstance(y, torch.Tensor):
                pass
            elif isinstance(y, list):
                y = torch.tensor(y)
            elif isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            else:
                raise TypeError('y should be an instance of list/np.ndarray/torch.Tensor, '
                                f'but got {type(y)}')
            return X.to(self.device), y.to(self.device)
        else:
            return X.to(self.device)

    def save_logs_to_tensorboard(self, saving_path):
        """ Save logs (self.logger) into a tensorboard file.

        Parameters
        ----------
        saving_path : str
            Local disk path to save the tensorboard file.
        """
        # TODO: find a solution for log saving
        raise IOError('This function is not ready for users.')
        # tb_summary_writer = SummaryWriter(saving_path)
        # tb_summary_writer.add_custom_scalars(self.logger)
        # tb_summary_writer.close()
        # print(f'Log saved successfully to {saving_path}.')

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

    def __init__(self, learning_rate, epochs, patience, batch_size, weight_decay, device):
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
        self.best_loss = float('inf')
        self.logger = {
            'training_loss': [],
            'validating_loss': []
        }

    def _print_model_size(self):
        """ Print the number of trainable parameters in the initialized NN model.
        """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model initialized successfully. Number of the trainable parameters: {num_params}')
