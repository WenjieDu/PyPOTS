"""
The core wrapper assembles the submodules of Koopa imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.koopa import BackboneKoopa
from ...nn.modules.saits import SaitsLoss, SaitsEmbedding


class _Koopa(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_seg_steps,
        d_dynamic,
        d_hidden,
        n_hidden_layers,
        n_blocks,
        multistep,
        alpha,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
        )
        self.backbone = BackboneKoopa(
            n_steps,
            n_features,
            n_steps,
            n_seg_steps,
            d_dynamic,
            d_hidden,
            n_hidden_layers,
            n_blocks,
            multistep,
            alpha,
        )

        # for the imputation task, the output dim is the same as input dim
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original Koopa paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)

        # Koopa encoder processing
        reconstruction = self.backbone(enc_out)

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
