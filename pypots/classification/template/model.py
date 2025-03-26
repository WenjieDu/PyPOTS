"""
The implementation of YourNewModel for the partially-observed time-series classification task.

TODO: modify the above description with your model's information.

"""

# Created by Your Name <Your contact email> TODO: modify the author information.
# License: BSD-3-Clause

from typing import Union, Optional

import torch

from .core import _YourNewModel

# TODO: import the base class from the classification package in PyPOTS.
#  Here I suppose this is a neural-network classification model.
#  You should make your model inherent BaseClassifier if it is not a NN.
# from ..base import BaseClassifier
from ..base import BaseNNClassifier
from ...nn.modules.loss import Criterion, CrossEntropy
from ...nn.modules.metric import PR_AUC
from ...optim.adam import Adam
from ...optim.base import Optimizer


# TODO: define your new model's wrapper here.
#  It should be a subclass of a base class defined in PyPOTS task packages (e.g.
#  BaseNNClassifier of PyPOTS classification task package). It has to implement all abstract methods of the base class.
#  Note that this class is a wrapper of your new model and will be directly exposed to users.
class YourNewModel(BaseNNClassifier):
    def __init__(
        self,
        # TODO: add your model's hyperparameters here
        n_classes: int,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = CrossEntropy,
        validation_metric: Union[Criterion, type] = CrossEntropy,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            n_classes=n_classes,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )
        # set up the hyperparameters
        # TODO: set up your model's hyperparameters here

        # set up the model
        self.model = _YourNewModel(
            # pass the arguments to your model
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()  # instantiate the optimizer if it is a class
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        raise NotImplementedError

    def _assemble_input_for_validating(self, data: list) -> dict:
        raise NotImplementedError

    def _assemble_input_for_testing(self, data: list) -> dict:
        raise NotImplementedError

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        raise NotImplementedError

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        raise NotImplementedError

    def classify(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> dict:
        raise NotImplementedError
