"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.autoformer import AutoformerEncoder
from ...nn.modules.saits import SaitsEmbedding


class _Autoformer(ModelCore):

    def __init__(
        self,
        n_classes: int,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        factor: int,
        moving_avg_window_size: int,
        dropout: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.d_model = d_model
        self.n_layers = n_layers
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=False,
            dropout=dropout,
        )
        self.encoder = AutoformerEncoder(
            n_layers,
            d_model,
            n_heads,
            d_ffn,
            factor,
            moving_avg_window_size,
            dropout,
            "relu",
        )
        self.projection = nn.Linear(d_model * n_steps, n_classes)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        enc_out = self.saits_embedding(X, missing_mask)

        # Autoformer encoder processing
        enc_out, attns = self.encoder(enc_out)
        logits = self.projection(enc_out.reshape(-1, self.n_steps * self.d_model))
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
