"""
The core wrapper assembles the submodules of TCN imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.tcn import BackboneTCN


class _TCN(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_levels: int,
        d_hidden: int,
        kernel_size: int,
        dropout: float = 0,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()
        self.n_steps = n_steps
        channel_sizes = [d_hidden] * n_levels

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
            dropout=dropout,
        )
        self.backbone = BackboneTCN(
            n_features,
            channel_sizes,
            kernel_size,
            dropout,
        )

        # for the imputation task, the output dim is the same as input dim
        self.output_projection = nn.Linear(channel_sizes[-1], n_features)
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original TCN paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)
        enc_out = enc_out.permute(0, 2, 1)

        # TCN encoder processing
        enc_out = self.backbone(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        # project back the original data space
        reconstruction = self.output_projection(enc_out)

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
