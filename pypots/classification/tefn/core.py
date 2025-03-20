"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.loss import Criterion
from ...nn.modules.loss import CrossEntropy
from ...nn.modules.saits import SaitsEmbedding
from ...nn.modules.tefn import BackboneTEFN


class _TEFN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_steps: int,
        n_features: int,
        n_fod: int,
        dropout: float,
        training_loss: Criterion = CrossEntropy(),
    ):
        super().__init__()

        self.n_fod = n_fod
        self.training_loss = training_loss

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
        self.activation_func = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(n_features * n_steps, n_classes)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        bz = X.shape[0]

        # WDU: the original FITS paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)

        # TEFN processing
        out = self.model(enc_out)
        out = self.activation_func(out)
        out = self.dropout(out)

        logits = self.output_projection(out.reshape(bz, -1))

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
