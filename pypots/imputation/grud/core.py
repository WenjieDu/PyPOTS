"""
The core wrapper assembles the submodules of GRU-D imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


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
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        # create models
        self.backbone = BackboneGRUD(
            n_steps,
            n_features,
            rnn_hidden_size,
        )
        self.output_projection = nn.Linear(rnn_hidden_size, n_features)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]
        deltas = inputs["deltas"]
        empirical_mean = inputs["empirical_mean"]
        X_filledLOCF = inputs["X_filledLOCF"]

        hidden_states, _ = self.backbone(X, missing_mask, deltas, empirical_mean, X_filledLOCF)

        # project back the original data space
        reconstruction = self.output_projection(hidden_states)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss = self.training_loss(reconstruction, X, missing_mask)
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
