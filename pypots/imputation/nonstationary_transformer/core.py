"""
The core wrapper assembles the submodules of NonstationaryTransformer imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.functional.normalization import nonstationary_norm, nonstationary_denorm
from ...nn.modules.nonstationary_transformer import (
    NonstationaryTransformerEncoder,
    Projector,
)
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _NonstationaryTransformer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        d_projector_hidden: list,
        n_projector_hidden_layers: int,
        dropout: float,
        attn_dropout: float,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        d_k = d_v = d_model // n_heads
        self.n_steps = n_steps

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            d_model,
            with_pos=True,
            dropout=dropout,
        )
        self.encoder = NonstationaryTransformerEncoder(
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )
        self.tau_learner = Projector(
            d_in=n_features,
            n_steps=n_steps,
            d_hidden=d_projector_hidden,
            n_hidden_layers=n_projector_hidden_layers,
            d_output=1,
        )
        self.delta_learner = Projector(
            d_in=n_features,
            n_steps=n_steps,
            d_hidden=d_projector_hidden,
            n_hidden_layers=n_projector_hidden_layers,
            d_output=n_steps,
        )

        # for the imputation task, the output dim is the same as input dim
        self.output_projection = nn.Linear(d_model, n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        X_enc, means, stdev = nonstationary_norm(X, missing_mask)

        tau = self.tau_learner(X, stdev).exp()
        delta = self.delta_learner(X, means)

        # WDU: the original Nonstationary Transformer paper isn't proposed for imputation task. Hence the model doesn't
        # take the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)

        # NonstationaryTransformer encoder processing
        enc_out, attns = self.encoder(enc_out, tau=tau, delta=delta)
        # project back the original data space
        reconstruction = self.output_projection(enc_out)
        reconstruction = nonstationary_denorm(reconstruction, means, stdev)

        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if self.training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            loss, ORT_loss, MIT_loss = self.saits_loss_func(reconstruction, X_ori, missing_mask, indicating_mask)
            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss

        return results
