"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsEmbedding
from ...nn.modules.transformer import TransformerEncoder


class _iTransformer(ModelCore):
    def __init__(
        self,
        n_classes: int,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_features = n_features
        self.d_model = d_model
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.saits_embedding = SaitsEmbedding(n_steps, d_model, with_pos=False, dropout=dropout)
        self.encoder = TransformerEncoder(
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model * n_features * 2, n_classes)

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        input_X = torch.cat([X.permute(0, 2, 1), missing_mask.permute(0, 2, 1)], dim=1)
        input_X = self.saits_embedding(input_X)
        bz = input_X.shape[0]

        # Transformer encoder processing
        enc_output, _ = self.encoder(input_X)
        enc_output = self.act(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = enc_output.reshape(bz, -1)
        logits = self.output_projection(enc_output)

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
