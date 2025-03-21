"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.loss import Criterion, CrossEntropy
from ...nn.modules.timesnet import BackboneTimesNet
from ...nn.modules.transformer.embedding import DataEmbedding


class _TimesNet(nn.Module):
    def __init__(
        self,
        n_classes,
        n_layers,
        n_steps,
        n_features,
        top_k,
        d_model,
        d_ffn,
        n_kernels,
        dropout,
        training_loss: Criterion = CrossEntropy(),
    ):
        super().__init__()

        self.n_steps = n_steps
        self.d_model = d_model
        self.n_layers = n_layers
        self.training_loss = training_loss

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
            n_max_steps=n_steps,
        )
        self.model = BackboneTimesNet(
            n_layers,
            n_steps,
            0,  # n_pred_steps should be 0 for the imputation task
            top_k,
            d_model,
            d_ffn,
            n_kernels,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * n_steps, n_classes)

    def forward(self, inputs: dict) -> dict:
        X = inputs["X"]

        # embedding
        input_X = self.enc_embedding(X)  # [B,T,C]
        # TimesNet processing
        enc_out = self.model(input_X)

        output = self.act(enc_out)
        output = self.dropout(output)
        logits = self.projection(output.reshape(-1, self.n_steps * self.d_model))

        classification_pred = torch.softmax(logits, dim=1)

        results = {
            "classification_pred": classification_pred,
            "logits": logits,
        }

        if self.training:
            # `loss` is always the item for backward propagating to update the model
            classification_loss = self.training_loss(logits, inputs["y"])
            results["loss"] = classification_loss

        return results
