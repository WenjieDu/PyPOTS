"""
The core wrapper assembles the submodules of SAITS classification model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn

from ...nn.modules.loss import Criterion, CrossEntropy
from ...nn.modules.saits import BackboneSAITS


class _SAITS(nn.Module):
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
        diagonal_attention_mask: bool = True,
        training_loss: Criterion = CrossEntropy(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_features = n_features
        self.diagonal_attention_mask = diagonal_attention_mask
        self.training_loss = training_loss
        # self.imputation_loss = MAE()

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
        classification_pred = torch.softmax(logits, dim=1)

        # ensemble the results as a dictionary for return
        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputed_data": imputed_data,
            "classification_pred": classification_pred,
            "logits": logits,
        }

        # if in training mode, return results with losses
        if self.training:
            # X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]

            # # calculate loss for the observed reconstruction task (ORT)
            # # this calculation is more complicated that pypots.nn.modules.saits.SaitsLoss because
            # # SAITS model structure has three parts of representation
            # ORT_loss = 0
            # ORT_loss += self.imputation_loss(X_tilde_1, X, missing_mask)
            # ORT_loss += self.imputation_loss(X_tilde_2, X, missing_mask)
            # ORT_loss += self.imputation_loss(X_tilde_3, X, missing_mask)
            # ORT_loss /= 3
            # ORT_loss = self.ORT_weight * ORT_loss
            #
            # # calculate loss for the masked imputation task (MIT)
            # MIT_loss = self.MIT_weight * self.imputation_loss(X_tilde_3, X_ori, indicating_mask)
            # # `loss` is always the item for backward propagating to update the model
            #
            # results["ORT_loss"] = ORT_loss
            # results["MIT_loss"] = MIT_loss

            classification_loss = self.training_loss(logits, inputs["y"])
            loss = classification_loss
            results["loss"] = loss

        return results
