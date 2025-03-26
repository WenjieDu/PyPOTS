"""
The core wrapper assembles the submodules of Raindrop classification model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.raindrop import BackboneRaindrop


class _Raindrop(ModelCore):
    def __init__(
        self,
        n_features,
        n_layers,
        d_model,
        n_heads,
        d_ffn,
        n_classes,
        dropout,
        max_len,
        d_static,
        aggregation: str,
        sensor_wise_mask: bool,
        static: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        d_pe = 16
        self.aggregation = aggregation
        self.sensor_wise_mask = sensor_wise_mask
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneRaindrop(
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_ffn,
            n_classes,
            dropout,
            max_len,
            d_static,
            d_pe,
            aggregation,
            sensor_wise_mask,
            static,
        )

        if static:
            d_final = d_model + n_features
        else:
            d_final = d_model + d_pe

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

    def forward(
        self,
        inputs,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask, static, timestamps, lengths = (
            inputs["X"],
            inputs["missing_mask"],
            inputs["static"],
            inputs["timestamps"],
            inputs["lengths"],
        )
        device = X.device
        batch_size = X.shape[1]

        representation, mask = self.backbone(
            X,
            timestamps,
            lengths,
        )

        lengths2 = lengths.unsqueeze(1).to(device)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if self.sensor_wise_mask:
            output = torch.zeros([batch_size, self.n_features, self.d_ob + 16], device=device)
            extended_missing_mask = missing_mask.view(-1, batch_size, self.n_features)
            for se in range(self.n_features):
                representation = representation.view(-1, batch_size, self.n_features, (self.d_ob + 16))
                out = representation[:, :, se, :]
                l_ = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)  # length
                out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (l_ + 1)
                output[:, se, :] = out_sensor
            output = output.view([-1, self.n_features * (self.d_ob + 16)])
        elif self.aggregation == "mean":
            output = torch.sum(representation * (1 - mask2), dim=0) / (lengths2 + 1)
        else:
            raise RuntimeError

        if static is not None:
            emb = self.static_emb(static)
            output = torch.cat([output, emb], dim=1)

        logits = self.mlp_static(output)
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
