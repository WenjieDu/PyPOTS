"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsEmbedding
from ...nn.modules.tefn import BackboneTEFN


class _TEFN(ModelCore):
    def __init__(
        self,
        n_classes: int,
        n_steps: int,
        n_features: int,
        n_fod: int,
        dropout: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_fod = n_fod
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
        )
        self.model = BackboneTEFN(
            n_steps,
            n_features,
            0,
            n_fod,
        )
        self.activation_func = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(n_features * n_steps, n_classes)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        bz = X.shape[0]

        enc_out = self.saits_embedding(X, missing_mask)

        # TEFN processing
        out = self.model(enc_out)
        out = self.activation_func(out)
        out = self.dropout(out)

        logits = self.output_projection(out.reshape(bz, -1))

        classification_proba = torch.softmax(logits, dim=1)

        results = {
            "classification_proba": classification_proba,
            "logits": logits,
        }

        if calc_criterion:
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = self.training_loss(logits, inputs["y"])
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(logits, inputs["y"])

        return results
