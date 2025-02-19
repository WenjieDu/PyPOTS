"""
The core wrapper assembles the submodules of BRITS classification model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.brits import BackboneBRITS


class _BRITS(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        classification_weight: float,
        reconstruction_weight: float,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        # create models
        self.model = BackboneBRITS(n_steps, n_features, rnn_hidden_size)
        self.f_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.b_classifier = nn.Linear(self.rnn_hidden_size, n_classes)

    def forward(self, inputs: dict) -> dict:
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
        classification_pred = (f_prediction + b_prediction) / 2

        results = {
            "imputed_data": imputed_data,
            "classification_pred": classification_pred,
        }

        # if in training mode, return results with losses
        if self.training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            f_classification_loss = F.nll_loss(torch.log(f_prediction), inputs["y"])
            b_classification_loss = F.nll_loss(torch.log(b_prediction), inputs["y"])
            classification_loss = (f_classification_loss + b_classification_loss) / 2
            loss = (
                consistency_loss
                + reconstruction_loss * self.reconstruction_weight
                + classification_loss * self.classification_weight
            )

            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss
            results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["classification_loss"] = classification_loss
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results
