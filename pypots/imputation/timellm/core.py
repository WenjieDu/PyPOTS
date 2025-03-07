"""
The core wrapper assembles the submodules of TimeLLM imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch.nn as nn

from ...nn.modules.saits import SaitsLoss
from ...nn.modules.timellm import BackboneTimeLLM


class _TimeLLM(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        patch_len,
        stride,
        d_model,
        d_ffn,
        d_llm,
        n_heads,
        llm_model_type,
        dropout,
        domain_prompt_content,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()

        self.n_steps = n_steps

        self.backbone = BackboneTimeLLM(
            n_steps,
            n_features,
            n_steps,
            n_layers,
            patch_len,
            stride,
            d_model,
            d_ffn,
            d_llm,
            n_heads,
            llm_model_type,
            dropout,
            domain_prompt_content,
            "imputation",
        )

        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # TimeLLM processing
        reconstruction = self.backbone(X, missing_mask)

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
