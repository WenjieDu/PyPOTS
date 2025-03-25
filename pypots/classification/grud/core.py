"""
The core wrapper assembles the submodules of GRU-D classification model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.grud import BackboneGRUD
from ...nn.modules.loss import Criterion


class _GRUD(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # create models
        self.model = BackboneGRUD(
            n_steps,
            n_features,
            rnn_hidden_size,
        )
        self.classifier = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs :
            The input data.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        _, hidden_state = self.model(X, missing_mask, deltas, empirical_mean, X_filledLOCF)

        logits = self.classifier(hidden_state)
        classification_proba = torch.softmax(logits, dim=1)

        results = {
            "logits": logits,
            "classification_proba": classification_proba,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                loss = self.training_loss(logits, inputs["y"])
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                metric = self.validation_metric(logits, inputs["y"])
                results["metric"] = metric

        return results
