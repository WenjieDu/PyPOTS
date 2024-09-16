
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


from .core import _BCSAI
from .data import DatasetForCSAI
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...optim.adam import Adam
from ...optim.base import Optimizer


class CSAI(BaseNNImputer):
    def __init__(self,
                 n_steps: int,
                 n_features: int,
                 rnn_hidden_size: int,
                 imputation_weight: float,
                 consistency_weight: float,
                 removal_percent: int,
                 increase_factor: float,
                 compute_intervals: bool,
                 step_channels:int,
                 batch_size: int, 
                 epochs: int, 
                 patience: Union[int, None ]= None, 
                 optimizer: Optional[Optimizer] = Adam(),
                 num_workers: int = 0, 
                 device: Union[str, torch.device, list, None ]= None, 
                 saving_path: str = None, 
                 model_saving_strategy: Union[str, None] = "best", 
                 verbose: bool = True):
        super().__init__(batch_size, epochs, patience, num_workers, device, saving_path, model_saving_strategy, verbose)
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.removal_percent = removal_percent
        self.increase_factor = increase_factor
        self.step_channels = step_channels
        self.compute_intervals = compute_intervals
        self.intervals = None
        
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
            back_last_obs
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
        }


        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
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
            X_ori,
            indicating_mask,
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
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }
        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)
    
    def fit(self, train_set, val_set, file_type: str = "hdf5",):
        
        self.training_set = DatasetForCSAI(
            train_set, False, False,
            file_type, self.removal_percent, 
            self.increase_factor, self.compute_intervals
        )

        training_loader = DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=collate_fn_bidirectional
        )

        if val_set is not None:
            # if not key_in_data_set("X_ori", val_set):
            #     raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForCSAI(
                val_set, False, False,  
                file_type, self.removal_percent, 
                self.increase_factor, self.compute_intervals,
                self.training_set.replacement_probabilities, 
                self.training_set.mean_set, self.training_set.std_set, True, False
            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                # collate_fn=collate_fn_bidirectional
            )

        # set up the model
        self.model = _BCSAI(self.n_steps, 
                            self.n_features, 
                            self.rnn_hidden_size, 
                            self.step_channels, 
                            self.training_set.intervals, 
                            self.consistency_weight, 
                            self.imputation_weight,
                            self.device)
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = self.optimizer
        self.optimizer.init_optimizer(self.model.parameters())

        # train the model
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=True)

    def predict(self, test_set: Union[dict, str], file_type: str = "hdf5") -> dict:
        
        if self.model == None:
            raise ValueError("Training must be run before predict")

        self.model.eval()
        test_set = DatasetForCSAI(
                test_set, False, False, 
                file_type, self.removal_percent, 
                self.increase_factor, self.compute_intervals,
                self.training_set.replacement_probabilities, 
                self.training_set.mean_set, self.training_set.std_set,True, False
            )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=collate_fn_bidirectional
        )
        imputation_collector = []
        x_ori_collector = []
        indicating_mask_collector = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputed_data = results["imputed_data"]
                imputation_collector.append(imputed_data)
                x_ori_collector.append(inputs["X_ori"])
                indicating_mask_collector.append(inputs["indicating_mask"])

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
            "X_ori": torch.cat(x_ori_collector).cpu().detach().numpy(),
            "indicating_mask": torch.cat(indicating_mask_collector).cpu().detach().numpy(),
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]

    