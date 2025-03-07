"""
The core wrapper assembles the submodules of TimeLLM forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from ...nn.functional.error import calc_mse
from ...nn.modules.timellm import BackboneTimeLLM


class _TimeLLM(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        term: str,
        n_layers: int,
        patch_len: int,
        stride: int,
        d_model: int,
        d_ffn: int,
        d_llm: int,
        n_heads: int,
        llm_model_type: str,
        dropout: float,
        domain_prompt_content: str,
    ):
        super().__init__()

        assert term in ["long", "short"], "forecasting term should be either 'long' or 'short'"
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features

        self.backbone = BackboneTimeLLM(
            n_steps,
            n_features,
            n_pred_steps,
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
            term + "_term_forecast",
        ).float()

    def forward(self, inputs: dict) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        if self.training:
            X_pred, X_pred_missing_mask = inputs["X_pred"], inputs["X_pred_missing_mask"]
        else:
            batch_size = X.shape[0]
            X_pred, X_pred_missing_mask = (
                torch.zeros(batch_size, self.n_pred_steps, self.n_pred_features),
                torch.ones(batch_size, self.n_pred_steps, self.n_pred_features),
            )

        # TimeLLM processing
        forecasting_result = self.backbone(X, missing_mask)
        # the raw output has length = n_steps+n_pred_steps, we only need the last n_pred_steps
        forecasting_result = forecasting_result[:, -self.n_pred_steps :]

        results = {
            "forecasting_data": forecasting_result,
        }

        # if in training mode, return results with losses
        if self.training:
            # `loss` is always the item for backward propagating to update the model
            results["loss"] = calc_mse(X_pred, forecasting_result, X_pred_missing_mask)

        return results
