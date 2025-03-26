"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from abc import abstractmethod

import torch.nn as nn


class ModelCore(nn.Module):
    """Base class for all NN models' core.
    For most of the NN models (inheriting BaseNNModel) in PyPOTS, their model core inherits this class and
    implement the forward and calc_criterion methods, and they exempt from overwriting _train_model() which
    carries concrete model training details, e.g. SAITS, Transformer, and this is task agnostic.
    But some models' processing procedures are too complex to be abstracted into this class, e.g. GAN models
    (like US-GAN and CRLI) which have more than one optimizer need to update or VaDER which needs pretraining and
    has complicated manipulations for its Gaussian mixture model, their model core don't have to inherit this class,
    just be a subclass of torch.nn.Module and obey basic rules (include the `loss`/`metric` as keys in the return dict)
    to be compatible with PyPOTS. Additionally, in this case, they need to overwrite _train_model() to implement
    their own training procedures.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        """Forward pass of the model.
        This method should be implemented to include all operations in the NN model's forward propagation.

        Parameters
        ----------
        inputs:
            The dictionary including all necessary input data for the model's forward pass.

        calc_criterion:
            Whether to calculate training loss (when self.training==True) and
            validation metric (when self.training!=True),
            and pass the criterion out together with the model forward results.

        Returns
        -------
        dict:
            The model's output results of its forward propagation.

        """
        raise NotImplementedError

    # @abstractmethod
    # def calc_criterion(self, inputs: dict) -> dict:
    #     """Calculate the loss and metric based on the model's forward pass results.
    #     This method should only be invoked in the training stage to calculate the training loss and validation metric.
    #     It won't be called during inference (i.e. predicting/testing stage) for cutting out unnecessary computation
    #     expenses, as the loss and metric are not needed in the inference stage.
    #
    #     Parameters
    #     ----------
    #     inputs:
    #         The dictionary including all necessary input data for the model's forward pass.
    #
    #     Returns
    #     -------
    #     dict:
    #         results bearing losses for and metrics
    #
    #     """
    #     # Usually, we run forward pass first to get the results of the model processing
    #
    #     # results = self.forward(inputs)
    #
    #     # Then, we calculate the loss (for training) and metric (for validation) based on the results below.
    #     # Please ENSURE the loss is returned as "loss" and the metric is returned as "metric" in the `results` dict.
    #     # PyPOTS will auto fetch the loss and metric from the returned dict to update and select the model.
    #
    #     # if self.training:  # if in the training mode (the training stage), return loss result from training_loss
    #     #     # `loss` is always the item for backward propagating to update the model
    #     #     loss =
    #     #     results["loss"] = loss
    #     # else:  # if in the eval mode (the validation stage), return metric result from validation_metric
    #     #
    #     #     metric =
    #     #     results["metric"] = metric
    #
    #     raise NotImplementedError
