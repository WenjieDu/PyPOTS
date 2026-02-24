"""
The core wrapper assembles the submodules of SeFT classification model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.seft import BackboneSeFT


class _SeFT(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ffn: int,
        n_seeds: int,
        dropout: float,
        max_timescale: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneSeFT(
            n_steps=n_steps,
            n_features=n_features,
            n_classes=n_classes,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ffn=d_ffn,
            n_seeds=n_seeds,
            dropout=dropout,
            max_timescale=max_timescale,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        logits = self.backbone(X, missing_mask)
        classification_proba = torch.softmax(logits, dim=1)

        results = {
            "logits": logits,
            "classification_proba": classification_proba,
        }

        if calc_criterion:
            if self.training:
                loss = self.training_loss(logits, inputs["y"])
                results["loss"] = loss
            else:
                metric = self.validation_metric(logits, inputs["y"])
                results["metric"] = metric

        return results
