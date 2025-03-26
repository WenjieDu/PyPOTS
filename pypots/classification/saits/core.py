"""
The core wrapper assembles the submodules of SAITS classification model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import BackboneSAITS


class _SAITS(ModelCore):
    def __init__(
        self,
        n_classes: int,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_features = n_features
        self.diagonal_attention_mask = diagonal_attention_mask
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.encoder = BackboneSAITS(
            n_steps,
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )

        self.classifier = nn.Linear(n_steps * n_features, n_classes)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
        diagonal_attention_mask: bool = True,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # determine the attention mask
        if (self.training and self.diagonal_attention_mask) or ((not self.training) and diagonal_attention_mask):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            # then broadcast on the batch axis
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None

        # SAITS processing
        (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(X, missing_mask, diagonal_attention_mask)

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_3

        logits = self.classifier(X_tilde_3.reshape(-1, self.n_steps * self.n_features))
        classification_proba = torch.softmax(logits, dim=1)

        # ensemble the results as a dictionary for return
        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputation": imputed_data,
            "classification_proba": classification_proba,
            "logits": logits,
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
