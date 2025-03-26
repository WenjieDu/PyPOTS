"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.csai import BackboneBCSAI
from ...nn.modules.loss import Criterion


class _BCSAI(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        imputation_weight: float,
        consistency_weight: float,
        classification_weight: float,
        n_classes: int,
        step_channels: int,
        dropout: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.classification_weight = classification_weight
        self.n_classes = n_classes
        self.step_channels = step_channels
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # create models
        self.model = BackboneBCSAI(
            n_steps,
            n_features,
            rnn_hidden_size,
            step_channels,
        )
        self.f_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.b_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

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

        f_logits = self.f_classifier(self.dropout(f_hidden_states))
        b_logits = self.b_classifier(self.dropout(b_hidden_states))
        f_prediction = torch.softmax(f_logits, dim=1)
        b_prediction = torch.softmax(b_logits, dim=1)
        classification_proba = (f_prediction + b_prediction) / 2

        results = {
            "imputation": imputed_data,
            "classification_proba": classification_proba,
            "f_logits": f_logits,
            "b_logits": b_logits,
            "consistency_loss": consistency_loss,
            "reconstruction_loss": reconstruction_loss,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                f_classification_loss = self.training_loss(f_logits, inputs["y"])
                b_classification_loss = self.training_loss(b_logits, inputs["y"])
                classification_loss = f_classification_loss + b_classification_loss
                loss = (
                    self.consistency_weight * consistency_loss
                    + self.imputation_weight * reconstruction_loss
                    + self.classification_weight * classification_loss
                )
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                f_validation_metric = self.validation_metric(f_logits, inputs["y"])
                b_validation_metric = self.validation_metric(b_logits, inputs["y"])
                validation_metric = (f_validation_metric + b_validation_metric) / 2
                results["metric"] = validation_metric

        return results
