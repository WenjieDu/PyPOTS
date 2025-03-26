"""
The core wrapper assembles the submodules of BRITS classification model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.brits import BackboneBRITS
from ...nn.modules.loss import Criterion


class _BRITS(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        classification_weight: float,
        reconstruction_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # create models
        self.model = BackboneBRITS(n_steps, n_features, rnn_hidden_size)
        self.f_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.b_classifier = nn.Linear(self.rnn_hidden_size, n_classes)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(inputs)

        f_logits = self.f_classifier(f_hidden_states)
        b_logits = self.b_classifier(b_hidden_states)
        f_prediction = torch.softmax(f_logits, dim=1)
        b_prediction = torch.softmax(b_logits, dim=1)
        classification_proba = (f_prediction + b_prediction) / 2

        results = {
            "imputation": imputed_data,
            "classification_proba": classification_proba,
            "consistency_loss": consistency_loss,
            "reconstruction_loss": reconstruction_loss,
            "f_logits": f_logits,
            "b_logits": b_logits,
            "f_reconstruction": f_reconstruction,
            "b_reconstruction": b_reconstruction,
        }

        if calc_criterion:
            f_classification_loss = self.training_loss(f_logits, inputs["y"])
            b_classification_loss = self.training_loss(b_logits, inputs["y"])
            classification_loss = (f_classification_loss + b_classification_loss) / 2
            loss = (
                consistency_loss
                + reconstruction_loss * self.reconstruction_weight
                + classification_loss * self.classification_weight
            )

            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                f_validation_metric = self.validation_metric(f_logits, inputs["y"])
                b_validation_metric = self.validation_metric(b_logits, inputs["y"])
                validation_metric = (f_validation_metric + b_validation_metric) / 2
                results["metric"] = validation_metric

        return results
