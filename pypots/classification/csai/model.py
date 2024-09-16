
from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _BCSAI
from .data import DatasetForCSAI
from ..base import BaseNNClassifier
from ...optim.adam import Adam
from ...optim.base import Optimizer


class CSAI(BaseNNClassifier):
    def __init__(self,
                 n_steps: int,
                 n_features: int,
                 rnn_hidden_size: int,
                 imputation_weight: float,
                 consistency_weight: float,
                 classification_weight: float,
                 n_classes: int,
                 removal_percent: int, 
                 increase_factor: float, 
                 compute_intervals: bool, 
                 step_channels:int, 
                 batch_size: int,  
                 epochs: int,  
                 dropout: float = 0.5,
                 patience: Union[int, None] = None,  
                 optimizer: Optimizer = Adam(), 
                 num_workers: int = 0,  
                 device: Optional[Union[str, torch.device, list]] = None,  
                 saving_path: str = None,
                 model_saving_strategy: Union[str, None] = "best", 
                 verbose: bool = True):
        super().__init__(
            n_classes, 
            batch_size, 
            epochs, 
            patience,
            num_workers, 
            device,
            saving_path, 
            model_saving_strategy, 
            verbose)
            
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.classification_weight = classification_weight
        self.removal_percent = removal_percent
        self.increase_factor = increase_factor
        self.step_channels = step_channels
        self.compute_intervals = compute_intervals
        self.intervals = None
        self.dropout = dropout
        self.device = device
        
        # Initialise empty model 
        self.model = None
        self.optimizer = optimizer

    def _assemble_input_for_training(self, data: list, training=True) -> dict:
        # extract data
        sample = data['sample']
        (
            indices,
            X,
            missing_mask,
            deltas,
            last_obs,
            back_X,
            back_missing_mask,
            back_deltas,
            back_last_obs,
            labels
        ) = self._send_data_to_given_device(sample)

        inputs = {
            "indices": indices,
            "labels": labels,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
                "last_obs": last_obs,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
                "last_obs": back_last_obs,
            },
        }
        

        return inputs
    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)
    
    def _assemble_input_for_testing(self, data: list) -> dict:
        # extract data
        sample = data['sample']
        (
            indices,
            X,
            missing_mask,
            deltas,
            last_obs,
            back_X,
            back_missing_mask,
            back_deltas,
            back_last_obs,
            # X_ori,
            # indicating_mask,
        ) = self._send_data_to_given_device(sample)

        # assemble input data
        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
                "last_obs": last_obs,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
                "last_obs": back_last_obs,
            },
            # "X_ori": X_ori,
            # "indicating_mask": indicating_mask,
        }
        
        return inputs
    
    def fit(
            self, 
            train_set,
            val_set=None,
            file_type: str = "hdf5",
    ):
        # Create dataset
        self.training_set = DatasetForCSAI(
            data=train_set,
            file_type=file_type,
            return_y=True,
            removal_percent=self.removal_percent,
            increase_factor=self.increase_factor,
            compute_intervals=self.compute_intervals,
        )

        train_loader = DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_loader = None
        if val_set is not None:
            val_set = DatasetForCSAI(
                data=val_set,
                file_type=file_type,
                return_y=True,
                removal_percent=self.removal_percent,
                increase_factor=self.increase_factor,
                compute_intervals=self.compute_intervals,
                replacement_probabilities=self.training_set.replacement_probabilities,
                normalise_mean=self.training_set.normalise_mean,
                normalise_std=self.training_set.normalise_std,
                training=False

            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        # Create model
        self.model = _BCSAI(
            n_steps=self.n_steps,
            n_features=self.n_features,
            rnn_hidden_size=self.rnn_hidden_size,
            imputation_weight=self.imputation_weight,
            consistency_weight=self.consistency_weight,
            classification_weight=self.classification_weight,
            n_classes=self.n_classes,
            step_channels=self.step_channels,
            dropout=self.dropout,
            intervals=self.training_set.intervals,
            device=self.device,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer.init_optimizer(self.model.parameters())    

        # train the model
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()

        self._auto_save_model_if_necessary(confirm_saving=True)


    def predict(
            self,
            test_set,
            file_type: str = "hdf5"):
        
        self.model.eval()
        test_set = DatasetForCSAI(
            data=test_set,
            file_type=file_type,
            return_y=False,
            removal_percent=self.removal_percent,
            increase_factor=self.increase_factor,
            compute_intervals=self.compute_intervals,
            replacement_probabilities=self.training_set.replacement_probabilities,
            normalise_mean=self.training_set.normalise_mean,
            normalise_std=self.training_set.normalise_std,
            training=False
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        classificaion_results = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                classificaion_results.append(results['classification_pred'])
            
        
        classification = torch.cat(classificaion_results).cpu().detach().numpy()
        result_dict = {
            "classification": classification,
        }  
        return result_dict
    
    def classify(
            self,
            test_set,
            file_type):
        
        result_dict = self.predict(test_set, file_type)
        return result_dict['classification']