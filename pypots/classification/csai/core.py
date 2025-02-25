"""

"""

# Created by Linglong Qian, Joseph Arul Raj <linglong.qian@kcl.ac.uk, joseph_arul_raj@kcl.ac.uk>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.csai import BackboneBCSAI


class _BCSAI(nn.Module):
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
        dropout: float = 0.5,
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

        # create models
        self.model = BackboneBCSAI(n_steps, n_features, rnn_hidden_size, step_channels)
        self.f_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.b_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

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

        f_logits = self.f_classifier(self.dropout(f_hidden_states))
        b_logits = self.b_classifier(self.dropout(b_hidden_states))
        f_prediction = torch.softmax(f_logits, dim=1)
        b_prediction = torch.softmax(b_logits, dim=1)
        classification_pred = (f_prediction + b_prediction) / 2

        results = {
            "imputed_data": imputed_data,
            "classification_pred": classification_pred,
        }

        # if in training mode, return results with losses
        if self.training:
            # criterion = DiceBCELoss().to(imputed_data.device)
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            f_classification_loss = F.nll_loss(torch.log(f_prediction), inputs["y"])
            b_classification_loss = F.nll_loss(torch.log(b_prediction), inputs["y"])
            classification_loss = f_classification_loss + b_classification_loss

            loss = (
                self.consistency_weight * consistency_loss
                + self.imputation_weight * reconstruction_loss
                + self.classification_weight * classification_loss
            )

            results["loss"] = loss
            results["classification_loss"] = classification_loss
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results
