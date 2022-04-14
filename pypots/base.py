"""
Base class for main models in PyPOTS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from abc import ABC, abstractmethod

import numpy as np
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


class BaseNNModel(BaseModel):
    """ Abstract class for all neural-network models.
    """

    def __init__(self, learning_rate, epochs, patience, batch_size, weight_decay, device):
        super(BaseNNModel, self).__init__(device)

        # training hype-parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
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

    @abstractmethod
    def input_data_processing(self, data):
        pass

    def _train_model(self, training_loader, val_loader=None):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss_collector, epoch_val_loss_collector = [], []
            for idx, data in enumerate(training_loader):
                inputs = self.input_data_processing(data)
                self.optimizer.zero_grad()
                results = self.model.forward(inputs)
                results['loss'].backward()
                self.optimizer.step()
                epoch_train_loss_collector.append(results['loss'].item())

            self.logger['training_loss'].extend(epoch_train_loss_collector)

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for idx, data in enumerate(val_loader):
                        inputs = self.input_data_processing(data)
                        results = self.model.forward(inputs)
                        epoch_val_loss_collector.append(results['loss'].item())
                self.logger['validating_loss'].extend(epoch_train_loss_collector)

            mean_train_loss = np.mean(epoch_train_loss_collector)  # mean training loss of the current epoch
            if val_loader is not None:
                mean_val_loss = np.mean(epoch_val_loss_collector)  # mean validating loss of the current epoch
                print(f'epoch {epoch}: training loss {mean_train_loss:.4f}, validating loss {mean_val_loss:.4f}')
                mean_loss = mean_val_loss
            else:
                print(f'epoch {epoch}: training loss {mean_train_loss:.4f}')
                mean_loss = mean_train_loss

            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.best_model_dict = self.model.state_dict()
            else:
                self.patience -= 1
                if self.patience == 0:
                    print('Exceeded the training patience. Terminating the training procedure...')
                    break
        print('Finished training.')
