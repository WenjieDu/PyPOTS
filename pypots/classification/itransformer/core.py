"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.loss import Criterion, CrossEntropy
from ...nn.modules.saits import SaitsEmbedding
from ...nn.modules.transformer import TransformerEncoder


class _iTransformer(nn.Module):
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
        training_loss: Criterion = CrossEntropy(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_features = n_features
        self.d_model = d_model
        self.training_loss = training_loss

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

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original Informer paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        input_X = torch.cat([X.permute(0, 2, 1), missing_mask.permute(0, 2, 1)], dim=1)
        input_X = self.saits_embedding(input_X)
        bz = input_X.shape[0]

        # Transformer encoder processing
        enc_output, _ = self.encoder(input_X)
        enc_output = self.act(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = enc_output.reshape(bz, -1)
        logits = self.output_projection(enc_output)

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
