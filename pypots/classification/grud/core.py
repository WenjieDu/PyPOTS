"""
The core wrapper assembles the submodules of GRU-D classification model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.grud import BackboneGRUD


class _GRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes

        # create models
        self.model = BackboneGRUD(
            n_steps,
            n_features,
            rnn_hidden_size,
        )
        self.classifier = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def forward(self, inputs: dict) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs :
            The input data.

        training :
            Whether in training mode.

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
        classification_pred = torch.softmax(logits, dim=1)
        results = {"classification_pred": classification_pred}

        # if in training mode, return results with losses
        if self.training:
            classification_loss = F.nll_loss(torch.log(classification_pred), inputs["y"])
            results["loss"] = classification_loss

        return results
