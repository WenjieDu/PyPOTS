"""
The core wrapper assembles the submodules of TKAN imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.saits import BackboneSAITS, SaitsEmbedding, SaitsLoss
from ...nn.modules.tkan import BackboneTKAN


class _TKAN(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_hidden: int,
        sub_kan_configs,
        sub_kan_output_dim,
        sub_kan_input_dim,
        dropout: float,
        ORT_weight: float,
        MIT_weight: float,
        training_loss: Criterion,
        validation_metric: Criterion,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features

        # SAITS-style embedding: concatenates X and missing_mask, projects to d_hidden
        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_hidden,
            with_pos=False,
        )

        # TKAN backbone
        self.backbone = BackboneTKAN(
            input_size=d_hidden,
            hidden_size=d_hidden,
            n_layers=n_layers,
            sub_kan_configs=sub_kan_configs,
            sub_kan_output_dim=sub_kan_output_dim,
            sub_kan_input_dim=sub_kan_input_dim,
            dropout=dropout,
        )

        # Output projection back to n_features
        self.output_projection = nn.Linear(d_hidden, n_features)

        # Apply SAITS loss (ORT + MIT)
        self.training_loss = SaitsLoss(ORT_weight, MIT_weight, training_loss)
        if validation_metric.__class__.__name__ == "Criterion":
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # Embed input: [X; missing_mask] -> d_hidden
        enc_out = self.saits_embedding(X, missing_mask)

        # TKAN backbone: [batch, n_steps, d_hidden] -> [batch, n_steps, d_hidden]
        enc_out = self.backbone(enc_out)

        # Project back to n_features
        reconstruction = self.output_projection(enc_out)

        # Combine observed values with imputed values
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            if self.training:
                loss, ORT_loss, MIT_loss = self.training_loss(
                    reconstruction, X_ori, missing_mask, indicating_mask
                )
                results["ORT_loss"] = ORT_loss
                results["MIT_loss"] = MIT_loss
                results["loss"] = loss
            else:
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results
