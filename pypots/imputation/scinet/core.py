"""
The core wrapper assembles the submodules of SCINet imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.saits import SaitsLoss, SaitsEmbedding
from ...nn.modules.scinet import BackboneSCINet


class _SCINet(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_stacks,
        n_levels,
        n_groups,
        n_decoder_layers,
        d_hidden,
        kernel_size,
        dropout,
        concat_len,
        pos_enc: bool,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
            dropout=dropout,
        )
        self.backbone = BackboneSCINet(
            n_out_steps=n_steps,
            n_in_steps=n_steps,
            n_in_features=n_features,
            d_hidden=d_hidden,
            n_stacks=n_stacks,
            n_levels=n_levels,
            n_decoder_layers=n_decoder_layers,
            n_groups=n_groups,
            kernel_size=kernel_size,
            dropout=dropout,
            concat_len=concat_len,
            modified=True,
            pos_enc=pos_enc,
            single_step_output_One=False,
        )

        # for the imputation task, the output dim is the same as input dim
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # WDU: the original SCINet paper isn't proposed for imputation task. Hence the model doesn't take
        # the missing mask into account, which means, in the process, the model doesn't know which part of
        # the input data is missing, and this may hurt the model's imputation performance. Therefore, I apply the
        # SAITS embedding method to project the concatenation of features and masks into a hidden space, as well as
        # the output layers to project back from the hidden space to the original space.
        enc_out = self.saits_embedding(X, missing_mask)

        # SCINet encoder processing
        reconstruction, _ = self.backbone(enc_out)

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
