"""
The core wrapper assembles the submodules of YourNewModel imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Your Name <Your contact email> TODO: modify the author information.
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules import ModelCore

# from ...nn.modules import some_modules
from ...nn.modules.loss import Criterion


# TODO: define your new model here.
#  It could be a neural network model or a non-neural network algorithm (e.g. written in numpy).
#  Your model should be implemented with PyTorch and subclass torch.nn.Module if it is a neural network.
#  Note that your main algorithm is defined in this class, and this class usually won't be exposed to users.
class _YourNewModel(ModelCore):
    def __init__(
        self,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # TODO: define your model's components here. If modules in pypots.nn.modules can be reused in your model,
        #  you can import them and use them here. AND if you think the modules you implemented can be reused by
        #  other models, you can also consider to contribute them to pypots.nn.modules
        self.embedding = nn.Module
        self.submodule = nn.Module
        self.backbone = nn.Module

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # TODO: define your model's forward propagation process here.
        #  The input is a dict, and the output `results` should also be a dict.
        reconstruction = self.backbone()  # replace this with your model's  process

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = self.training_loss(reconstruction, X_ori, indicating_mask)
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
