"""
The core wrapper assembles the submodules of Transformer imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.transformer import TransformerEncoder


class _Transformer(ModelCore):
    def __init__(
        self,
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
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            n_max_steps=n_steps,
            dropout=dropout,
        )
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
        self.output_projection = nn.Linear(d_model, n_features)

        # apply SAITS loss function to Transformer on the imputation task
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            # in this case, we need validation_metric.lower_better in _train_model() so only pass Criterion()
            # we use training_loss as validation_metric for concrete calculation process
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # apply the SAITS embedding strategy, concatenate X and missing mask for input
        input_X = self.saits_embedding(X, missing_mask)

        # Transformer encoder processing
        enc_output, _ = self.encoder(input_X)
        # project the representation from the d_model-dimensional space to the original data space for output
        reconstruction = self.output_projection(enc_output)

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        # ensemble the results as a dictionary for return
        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:  # if in the training mode (the training stage), return loss result from training_loss
                # `loss` is always the item for backward propagating to update the model
                loss, ORT_loss, MIT_loss = self.training_loss(reconstruction, X_ori, missing_mask, indicating_mask)
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                # `loss` is always the item for backward propagating to update the model
                results["loss"] = loss
            else:  # if in the eval mode (the validation stage), return metric result from validation_metric
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
